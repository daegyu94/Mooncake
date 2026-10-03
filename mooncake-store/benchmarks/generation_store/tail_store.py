"""Generation-scoped write combining for recomputable KV, not durable PUT.

append() accepts into the owned RAM/DFS cache. flush() is the explicit DFS
completion barrier. invalidate() revokes new readers; a pinned old reader
retains its tail until release. The parent prototype's single-owner scope,
fail-closed restart, and synchronous native completion remain unchanged.
"""

from generation_store import GenerationStore


class TailBufferedStore(GenerationStore):
    def __init__(self, io, root, capacity, chunk_bytes=1 << 20, tail_budget=4 << 20):
        if chunk_bytes <= 0 or tail_budget < 0:
            raise ValueError("invalid chunk or tail budget")
        super().__init__(io, root, capacity)
        self.chunk_bytes = chunk_bytes
        self.tail_budget = tail_budget
        self.resident_tail_bytes = 0
        self.sequence = 0
        self.metrics.update(
            accepted_bytes=0,
            flushed_bytes=0,
            discarded_tail_bytes=0,
            pressure_flush_bytes=0,
            resident_peak_bytes=0,
            transient_peak_bytes=0,
            explicit_tail_copy_bytes=0,
            tail_read_bytes=0,
            flush_calls=0,
        )

    def create(self):
        with self.lock:
            gid = super().create()
            g = self.generations[gid]
            g.tail = bytearray()
            g.persisted = 0
            g.touch = 0
            return gid

    def _flush(self, g, length, pressure=False):
        if not length:
            return
        assert 0 < length <= len(g.tail)
        payload = bytes(memoryview(g.tail)[:length])
        self.metrics["explicit_tail_copy_bytes"] += length
        ranges = [
            (g.persisted + i, payload[i : i + self.io.slot_bytes])
            for i in range(0, length, self.io.slot_bytes)
        ]
        try:
            self.io.io(g.fd, ranges)
        except Exception:
            # A partial native failure cannot publish a successful cache hit.
            g.state = "failed"
            raise
        g.persisted += length
        del g.tail[:length]
        self.resident_tail_bytes -= length
        self.metrics["flushed_bytes"] += length
        self.metrics["flush_calls"] += 1
        if pressure:
            self.metrics["pressure_flush_bytes"] += length

    def _enforce_budget(self):
        # Retired pinned readers cannot lose their volatile bytes. Spill live
        # tails instead; this may create partial-chunk writes and is measured.
        while self.resident_tail_bytes > self.tail_budget:
            candidates = [
                g
                for g in self.generations.values()
                if g.tail and g.state in ("active", "sealed")
            ]
            if not candidates:
                raise BufferError("pinned retired tails exhaust budget")
            victim = min(candidates, key=lambda g: g.touch)
            self._flush(victim, len(victim.tail), pressure=True)
        self.metrics["resident_peak_bytes"] = max(
            self.metrics["resident_peak_bytes"], self.resident_tail_bytes
        )

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
                    g.tail.extend(value)
                    self.resident_tail_bytes += len(value)
                    self.metrics["explicit_tail_copy_bytes"] += len(value)
                    self.metrics["transient_peak_bytes"] = max(
                        self.metrics["transient_peak_bytes"], self.resident_tail_bytes
                    )
                    end = g.persisted + len(g.tail)
                    full = end // self.chunk_bytes * self.chunk_bytes - g.persisted
                    self._flush(g, max(0, full))
                    self.sequence += 1
                    g.touch = self.sequence
                    self._enforce_budget()
                # Only a completely accepted append publishes its logical keys.
                for value in values:
                    g.append_index(g.cursor, len(value))
                    g.cursor += len(value)
            except Exception:
                g.state = "failed"
                # A failed multi-value append can already own disk/tail bytes.
                # Reserve them until invalidate+GC even without published keys.
                g.cursor = max(g.cursor, g.persisted + len(g.tail))
                raise
            self.metrics["accepted_bytes"] += length
            return list(range(old_count, g.count))

    def seal(self, gid):
        with self.lock:
            g = self.generations[gid]
            if g.state not in ("active", "sealed"):
                raise ValueError("cannot seal failed generation")
            g.state = "sealed"

    def flush(self, gid):
        with self.lock:
            g = self.generations[gid]
            if g.state not in ("active", "sealed"):
                raise ValueError("cannot flush failed generation")
            self._flush(g, len(g.tail))

    def acquire(self, gid):
        with self.lock:
            if self.generations[gid].state == "failed":
                raise ValueError("failed generation is not readable")
            return super().acquire(gid)

    def read_lease(self, g, keys, channel=None):
        with self.lock:
            if g.readers <= 0 or g.state == "failed":
                raise ValueError("read requires a valid reader lease")
            plans, ranges = [], []
            for key in keys:
                off, size = g.location(key)
                end = off + size
                disk_end = min(end, g.persisted)
                first = len(ranges)
                for pos in range(off, disk_end, self.io.slot_bytes):
                    ranges.append((pos, min(self.io.slot_bytes, disk_end - pos)))
                ram_start = max(off, g.persisted)
                tail = (
                    bytes(
                        memoryview(g.tail)[ram_start - g.persisted : end - g.persisted]
                    )
                    if ram_start < end
                    else b""
                )
                plans.append((first, len(ranges), tail))
                self.metrics["tail_read_bytes"] += len(tail)
            # Snapshot tail before releasing the lock. Append/flush can now
            # advance the frontier; immutable disk bytes and lease pin are safe.
        data = (channel or self.io).io(g.fd, ranges, read=True) if ranges else []
        return [b"".join(data[first:last] + [tail]) for first, last, tail in plans]

    def get(self, gid, keys, channel=None):
        lease = self.acquire(gid)
        try:
            return self.read_lease(lease, keys, channel)
        finally:
            self.release(lease)

    def _discard_tail(self, g):
        self.metrics["discarded_tail_bytes"] += len(g.tail)
        self.resident_tail_bytes -= len(g.tail)
        g.tail.clear()

    def invalidate(self, gid, individual=False):
        with self.lock:
            if individual:
                raise ValueError("tail store uses generation retirement")
            super().invalidate(gid)
            g = self.retired[gid]
            if not g.readers:
                self._discard_tail(g)

    def release(self, g):
        with self.lock:
            super().release(g)
            if not g.readers and g.state == "retired":
                self._discard_tail(g)
