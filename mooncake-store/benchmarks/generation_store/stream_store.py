"""Segment-owned write combining plus exact-adjacency native read scatter.

The no-gap planner is derived from research/native-locality-planner commit
1eac5b07f30e2ddcaf9972045254a5119dd74eab. Adjacency is prior work, not the new
mechanism. A/B use the same plan; B replaces mutable tail/staging/scatter copies.
"""

import time
from collections import deque

from tail_store import TailBufferedStore
from vector_io import allocate_output


def plan_adjacent(locations, max_span):
    """No gaps and bounded spans; preserve key order and duplicate outputs."""
    if max_span <= 0 or any(off < 0 or size < 0 for off, size in locations):
        raise ValueError("invalid range geometry")
    unique = sorted({(off, size) for off, size in locations if size})
    ranges = []
    previous_end = -1
    for off, size in unique:
        if off < previous_end:
            raise ValueError("overlapping index entries")
        previous_end = off + size
        end = off + size
        while off < end:
            if (
                ranges
                and ranges[-1][0] + ranges[-1][1] == off
                and ranges[-1][1] < max_span
            ):
                start, length = ranges[-1]
                take = min(end - off, max_span - length)
                ranges[-1] = (start, length + take)
            else:
                take = min(end - off, max_span)
                ranges.append((off, take))
            off += take
    # Mapping costs scale with index entries/fragments, not bytes.
    from bisect import bisect_right

    starts = [off for off, _ in ranges]
    pieces = []
    for off, size in locations:
        spans, destination, end = [], 0, off + size
        while off < end:
            i = bisect_right(starts, off) - 1
            begin, length = ranges[i]
            take = min(end - off, begin + length - off)
            if take <= 0:
                raise ValueError("uncovered index entry")
            spans.append((i, off - begin, take, destination))
            destination += take
            off += take
        pieces.append(spans)
    return ranges, pieces, sum(size for _, size in unique)


class SegmentTail:
    def __init__(self):
        self.parts = deque()
        self.length = 0

    def __len__(self):
        return self.length

    def append(self, value):
        self.parts.append((value, 0, len(value)))
        self.length += len(value)

    def slice(self, start, length):
        if start < 0 or length < 0 or start + length > self.length:
            raise ValueError("tail slice out of bounds")
        result = []
        for value, begin, size in self.parts:
            if start >= size:
                start -= size
                continue
            take = min(size - start, length)
            if take:
                result.append((value, begin + start, take))
            length -= take
            if not length:
                break
            start = 0
        return result

    def consume(self, size, io):
        """Compact only a live suffix so consumed backing bytes are not retained."""
        self.length -= size
        compacted = 0
        while size:
            value, begin, length = self.parts.popleft()
            if size < length:
                remaining = length - size
                value = io.copy_parts([(value, begin + size, remaining)])
                self.parts.appendleft((value, 0, remaining))
                compacted += remaining
                break
            size -= length
        assert sum(len(value) for value, _, _ in self.parts) == self.length
        return compacted

    def clear(self):
        self.parts.clear()
        self.length = 0


class PlannedTailStore(TailBufferedStore):
    """Strong A: unchanged mutable tail with the identical adjacency layout."""

    def _capture(self, g, keys, io):
        with self.lock:
            if g.readers <= 0 or g.state == "failed":
                raise ValueError("read requires a valid reader lease")
            locations = [g.location(key) for key in keys]
            disk = [
                (off, max(0, min(off + size, g.persisted) - off))
                for off, size in locations
            ]
            ranges, pieces, unique = plan_adjacent(disk, io.slot_bytes)
            return locations, disk, ranges, pieces, unique

    def read_lease(self, g, keys, channel=None):
        io = channel or self.io
        with self.lock:
            start = time.thread_time_ns()
            locations, _disk, ranges, pieces, unique = self._capture(g, keys, io)
            tails = [
                bytes(
                    memoryview(g.tail)[
                        max(off, g.persisted) - g.persisted : off + size - g.persisted
                    ]
                )
                if off + size > g.persisted
                else b""
                for off, size in locations
            ]
            self.metrics["plan_cpu_ns"] = (
                self.metrics.get("plan_cpu_ns", 0) + time.thread_time_ns() - start
            )
            self.metrics["tail_read_bytes"] += sum(map(len, tails))
            self.metrics["read_tail_snapshot_copy_bytes"] = self.metrics.get(
                "read_tail_snapshot_copy_bytes", 0
            ) + sum(map(len, tails))
            # Reserve only after the snapshot is complete; an allocation error
            # before this point cannot leak an operation reader reference.
            g.readers += 1
        try:
            buffers = io.io(g.fd, ranges, read=True) if ranges else []
            start = time.thread_time_ns()
            result, copied = [], 0
            for spans, tail in zip(pieces, tails):
                parts = []
                for index, off, size, _ in spans:
                    value = buffers[index]
                    parts.append(value[off : off + size])
                    if off or size != len(value):
                        copied += size
                parts.append(tail)
                result.append(b"".join(parts))
                if len(parts) > 1:
                    copied += sum(map(len, parts))
            with self.lock:
                self.metrics["scatter_cpu_ns"] = (
                    self.metrics.get("scatter_cpu_ns", 0)
                    + time.thread_time_ns()
                    - start
                )
                self.metrics["read_scatter_copy_bytes"] = (
                    self.metrics.get("read_scatter_copy_bytes", 0) + copied
                )
                self.metrics["planned_ranges"] = self.metrics.get(
                    "planned_ranges", 0
                ) + len(ranges)
                self.metrics["unique_disk_requested_bytes"] = (
                    self.metrics.get("unique_disk_requested_bytes", 0) + unique
                )
            return result
        finally:
            self.release(g)


class VectorTailStore(PlannedTailStore):
    """B: immutable owned segments -> registered slots -> final owned outputs."""

    def __init__(self, io, root, capacity, chunk_bytes=1 << 20, tail_budget=4 << 20):
        super().__init__(io, root, capacity, chunk_bytes, tail_budget)
        self.metrics.update(ownership_copy_bytes=0, suffix_compact_copy_bytes=0)

    def create(self):
        gid = super().create()
        with self.lock:
            self.generations[gid].tail = SegmentTail()
        return gid

    def _flush(self, g, length, pressure=False):
        if not length:
            return
        assert 0 < length <= len(g.tail)
        ranges = [
            (
                g.persisted + start,
                g.tail.slice(start, min(self.io.slot_bytes, length - start)),
            )
            for start in range(0, length, self.io.slot_bytes)
        ]
        try:
            self.io.write_parts(g.fd, ranges)
        except Exception:
            g.state = "failed"
            raise
        g.persisted += length
        self.resident_tail_bytes -= length
        self.metrics["flushed_bytes"] += length
        self.metrics["flush_calls"] += 1
        if pressure:
            self.metrics["pressure_flush_bytes"] += length
        compacted = g.tail.consume(length, self.io)
        self.metrics["suffix_compact_copy_bytes"] += compacted

    def append(self, gid, values):
        if not values or any(not value for value in values):
            raise ValueError("append requires nonempty values")
        with self.lock:
            g = self.generations[gid]
            if g.state != "active":
                raise ValueError("late PUT to non-active generation")
            length = sum(map(len, values))
            used = sum(
                x.cursor
                for x in list(self.generations.values()) + list(self.retired.values())
            )
            if used + length > self.capacity:
                self.metrics["capacity_rejections"] += 1
                raise BufferError("capacity includes pinned retired generations")
            old_count = g.count
            try:
                for value in values:
                    # Immutable ownership is a retained reference; mutable
                    # caller buffers require a real defensive copy.
                    owned = value if type(value) is bytes else bytes(value)
                    if owned is not value:
                        self.metrics["ownership_copy_bytes"] += len(owned)
                    g.tail.append(owned)
                    self.resident_tail_bytes += len(owned)
                    self.metrics["transient_peak_bytes"] = max(
                        self.metrics["transient_peak_bytes"], self.resident_tail_bytes
                    )
                    end = g.persisted + len(g.tail)
                    full = end // self.chunk_bytes * self.chunk_bytes - g.persisted
                    self._flush(g, max(0, full))
                    self.sequence += 1
                    g.touch = self.sequence
                    self._enforce_budget()
                for value in values:
                    g.append_index(g.cursor, len(value))
                    g.cursor += len(value)
            except Exception:
                g.state = "failed"
                g.cursor = max(g.cursor, g.persisted + len(g.tail))
                raise
            self.metrics["accepted_bytes"] += length
            return list(range(old_count, g.count))

    def read_lease(self, g, keys, channel=None):
        io = channel or self.io
        with self.lock:
            start = time.thread_time_ns()
            locations, disk, ranges, pieces, unique = self._capture(g, keys, io)
            tails = [
                g.tail.slice(
                    max(off, g.persisted) - g.persisted,
                    off + size - max(off, g.persisted),
                )
                if off + size > g.persisted
                else []
                for off, size in locations
            ]
            self.metrics["plan_cpu_ns"] = (
                self.metrics.get("plan_cpu_ns", 0) + time.thread_time_ns() - start
            )
            self.metrics["tail_read_bytes"] += sum(
                length for parts in tails for _, _, length in parts
            )
            g.readers += 1
        try:
            outputs = [allocate_output(size) for _, size in locations]
            io.read_into(g.fd, ranges, pieces, outputs)
            start = time.thread_time_ns()
            fragments = []
            for output, parts, (_, disk_size) in zip(outputs, tails, disk):
                destination = disk_size
                for value, begin, size in parts:
                    fragments.append((value, begin, size, output, destination))
                    destination += size
            io.copy_into(fragments)
            with self.lock:
                self.metrics["scatter_cpu_ns"] = (
                    self.metrics.get("scatter_cpu_ns", 0)
                    + time.thread_time_ns()
                    - start
                )
                self.metrics["planned_ranges"] = self.metrics.get(
                    "planned_ranges", 0
                ) + len(ranges)
                self.metrics["unique_disk_requested_bytes"] = (
                    self.metrics.get("unique_disk_requested_bytes", 0) + unique
                )
            return outputs
        finally:
            self.release(g)
