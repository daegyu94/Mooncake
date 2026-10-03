"""Generation-local range coalescing with a hard span and over-read budget.

The index and reader lease remain owned by GenerationStore. The planner neither
mixes generations nor changes publication/reclaim semantics. This is a native
research path, not a production Mooncake DFS protocol implementation.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math
import time

from generation_store import GenerationStore


@dataclass
class ReadPlan:
    ranges: list
    pieces: list
    unique_bytes: int

    def scatter(self, buffers):
        if len(buffers) != len(self.ranges):
            raise ValueError("completion count mismatch")
        for buf, (_, size) in zip(buffers, self.ranges):
            if len(buf) != size:
                raise ValueError("short completion")
        output, copy_bytes = [], 0
        for spans in self.pieces:
            parts = []
            for index, offset, size in spans:
                buf = buffers[index]
                parts.append(buf[offset : offset + size])
                if offset or size != len(buf):
                    copy_bytes += size
            output.append(b"".join(parts))
            if len(parts) > 1:
                copy_bytes += sum(map(len, parts))
        return output, copy_bytes


def plan_ranges(locations, max_span, amplification=1.0):
    """Greedy bounded coalescing, preserving order/duplicates at scatter.

    Locations are immutable, non-overlapping index entries (duplicates allowed).
    The amplification denominator is unique requested bytes, not duplicate GETs.
    Every merged range independently satisfies the byte budget. A rejected gap
    is not revisited after later entries: this is deliberately not an optimizer.
    """
    if max_span <= 0 or not math.isfinite(amplification) or amplification < 1:
        raise ValueError("invalid range budget")
    locations = list(locations)
    if any(off < 0 or size < 0 for off, size in locations):
        raise ValueError("negative index location")
    unique = sorted(set((off, size) for off, size in locations if size))
    previous_end = -1
    fragments = []
    for off, size in unique:
        if off < previous_end:
            raise ValueError("overlapping index entries")
        previous_end = off + size
        for delta in range(0, size, max_span):
            fragments.append((off + delta, min(max_span, size - delta)))
    merged = []
    useful = 0
    for off, size in fragments:
        if merged:
            start, length = merged[-1]
            span = off + size - start
            if span <= max_span and span <= amplification * (useful + size):
                merged[-1] = (start, span)
                useful += size
                continue
        merged.append((off, size))
        useful = size
    starts = [off for off, _ in merged]
    pieces = []
    for off, size in locations:
        end, spans = off + size, []
        while off < end:
            i = bisect_right(starts, off) - 1
            start, length = merged[i]
            count = min(end - off, start + length - off)
            if count <= 0:
                raise ValueError("uncovered index entry")
            spans.append((i, off - start, count))
            off += count
        pieces.append(spans)
    return ReadPlan(merged, pieces, sum(size for _, size in unique))


class PlannedStore(GenerationStore):
    def get_planned(self, gid, keys, channel=None, amplification=1.0):
        io = channel or self.io
        g = self.acquire(gid)
        try:
            start = time.thread_time_ns()
            with self.lock:
                locations = [g.location(key) for key in keys]
            plan = plan_ranges(locations, io.slot_bytes, amplification)
            plan_ns = time.thread_time_ns() - start
            buffers = io.io(g.fd, plan.ranges, read=True)
            start = time.thread_time_ns()
            values, copies = plan.scatter(buffers)
            scatter_ns = time.thread_time_ns() - start
            with self.lock:
                for key, value in {
                    "plan_cpu_ns": plan_ns,
                    "scatter_cpu_ns": scatter_ns,
                    "scatter_copy_bytes": copies,
                    "planned_ranges": len(plan.ranges),
                    "unique_requested_bytes": plan.unique_bytes,
                }.items():
                    self.metrics[key] = self.metrics.get(key, 0) + value
            return values
        finally:
            self.release(g)
