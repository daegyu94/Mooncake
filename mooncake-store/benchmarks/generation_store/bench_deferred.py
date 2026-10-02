"""128 epochs x 5 pairs: moving reclaim off transition can grow retired bytes."""

import argparse
import json
from pathlib import Path
import time
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
for repeat in range(5):
    for cadence in [1, 8] if repeat % 2 == 0 else [8, 1]:
        s = GenerationStore(
            io, a.mount / "verl-lab" / ("deferred-" + uuid.uuid4().hex), 8 << 20
        )
        events = []
        before = dict(io.stats)

        def workload():
            for epoch in range(128):
                g = s.create()
                payload = bytes([epoch]) * (64 << 10)
                s.append(g, [payload] * 8)
                s.seal(g)
                assert s.get(g, [0]) == [payload]
                t = time.perf_counter_ns()
                s.invalidate(g)
                if cadence == 1:
                    s.gc()
                transition = time.perf_counter_ns() - t
                debt = sum(x.cursor for x in s.retired.values())
                events.append(
                    dict(
                        epoch=epoch,
                        transition_ns=transition,
                        retired_bytes=debt,
                        fd_count=len(list(Path("/proc/self/fd").iterdir())),
                    )
                )
                if cadence != 1 and (epoch + 1) % cadence == 0:
                    s.gc()
            s.gc()

        _, measurement = timed(workload)
        assert not s.retired and not s.generations
        rows.append(
            dict(
                repeat=repeat,
                cadence=cadence,
                events=events,
                **measurement,
                logical_write_bytes=128 * 512 * 1024,
                metrics=s.metrics,
                io_delta={k: io.stats[k] - before[k] for k in io.stats},
            )
        )
        (a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(repeat, cadence, measurement, flush=True)
io.close()
