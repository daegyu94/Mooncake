"""Native pressure/mixed scenarios and isolated index destruction costs."""

import argparse
import array
import json
from pathlib import Path
import sys
import uuid
from generation_store import GenerationStore, NativeIO
from bench_generation import timed

p = argparse.ArgumentParser()
p.add_argument("--library", type=Path, required=True)
p.add_argument("--mount", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
io = NativeIO(a.library, a.mount, depth=8)
rows = []
for ratio in [0, 1, 4]:
    for repeat in range(5):
        for individual in [True, False] if repeat % 2 == 0 else [False, True]:
            s = GenerationStore(
                io,
                a.mount / "verl-lab" / ("mix-" + uuid.uuid4().hex),
                2 << 20,
                compact=not individual,
            )
            g = s.create()
            payload = bytes([repeat]) * (64 << 10)
            before = dict(io.stats)

            def work():
                for burst in range(8):
                    keys = s.append(g, [payload] * 4)
                    for _ in range(ratio):
                        assert s.get(g, keys) == [payload] * 4

            _, m = timed(work)
            s.seal(g)
            _, inv = timed(lambda: s.invalidate(g, individual=individual))
            _, gc = timed(s.gc)
            rows.append(
                dict(
                    reuse_ratio=ratio,
                    repeat=repeat,
                    individual=individual,
                    **m,
                    invalidate_ns=inv["wall_ns"],
                    gc_ns=gc["wall_ns"],
                    logical_put_bytes=2 << 20,
                    read_bytes=io.stats["read_bytes"] - before["read_bytes"],
                    metadata=s.metrics,
                )
            )
(a.output / "mixed.json").write_text(json.dumps(rows, indent=2) + "\n")
pressure = []
for cap in [1 << 20, 2 << 20, 8 << 20]:
    for repeat in range(5):
        s = GenerationStore(
            io, a.mount / "verl-lab" / ("pressure-" + uuid.uuid4().hex), cap
        )
        g = s.create()
        s.append(g, [b"a" * (512 << 10)])
        s.seal(g)
        lease = s.acquire(g)
        s.invalidate(g)
        n = s.create()
        rejected = False
        try:
            s.append(n, [b"b" * (768 << 10)])
        except BufferError:
            rejected = True
        assert rejected == (cap < (1280 << 10))
        assert s.gc() == 0
        s.release(lease)
        assert s.gc() == 1
        if rejected:
            s.append(n, [b"b" * (768 << 10)])
        assert s.get(n, [0]) == [b"b" * (768 << 10)]
        s.invalidate(n)
        s.gc()
        pressure.append(
            dict(capacity=cap, repeat=repeat, rejected=rejected, metrics=s.metrics)
        )
(a.output / "pressure.json").write_text(json.dumps(pressure, indent=2) + "\n")
# Two indexes with identical ordinals and mappings. De-duplicate shared ints.
index = []
for count in [1024, 65536]:
    for compact in [False, True]:
        for repeat in range(5):
            idx, build = timed(
                lambda: array.array(
                    "Q", [v for i in range(count) for v in (i * 458752, 458752)]
                )
                if compact
                else {i: (i * 458752, 458752) for i in range(count)}
            )
            seen = set()

            def footprint(value):
                if id(value) in seen:
                    return 0
                seen.add(id(value))
                size = sys.getsizeof(value)
                if isinstance(value, dict):
                    size += sum(footprint(k) + footprint(v) for k, v in value.items())
                elif isinstance(value, tuple):
                    size += sum(footprint(v) for v in value)
                return size

            mem = footprint(idx)
            result, lookup = timed(
                lambda: sum(
                    idx[2 * i] if compact else idx[i][0] for i in range(0, count, 17)
                )
            )
            assert result == sum(i * 458752 for i in range(0, count, 17))
            _, destroy = timed(
                lambda: idx.__delitem__(slice(None)) if compact else idx.clear()
            )
            index.append(
                dict(
                    count=count,
                    compact=compact,
                    repeat=repeat,
                    footprint=mem,
                    build_cpu_ns=build["cpu_ns"],
                    lookup_cpu_ns=lookup["cpu_ns"],
                    destroy_cpu_ns=destroy["cpu_ns"],
                )
            )
(a.output / "index-final.json").write_text(json.dumps(index, indent=2) + "\n")
io.close()
print("Native mixed/pressure and index A/B complete", flush=True)
