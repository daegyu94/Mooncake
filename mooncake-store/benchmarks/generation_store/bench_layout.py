"""Controlled append-granularity comparison with real server syscall evidence."""

import argparse
import importlib
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
p.add_argument("--probe-module", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
sys.path.insert(0, str(a.probe_module))
StorageIO = importlib.import_module("syscall_io").StorageIO

owner = NativeIO(a.library, a.mount, depth=8)
s = GenerationStore(
    owner, a.mount / "verl-lab" / ("layout-" + uuid.uuid4().hex), 1 << 30
)
values = [bytes([i]) * (448 << 10) for i in range(128)]
rows = []
for repeat in range(5):
    for packed in [False, True] if repeat % 2 == 0 else [True, False]:
        name = f"r{repeat}-" + ("packed" if packed else "objects")
        out = a.output / name
        out.mkdir()
        g = s.create()
        start = dict(owner.stats)
        with StorageIO(out):
            _, measure = timed(
                lambda: s.append(g, values)
                if packed
                else [s.append(g, [v]) for v in values]
            )
        assert s.get(g, [0, 127]) == [values[0], values[-1]]
        rows.append(
            dict(
                name=name,
                packed=packed,
                repeat=repeat,
                logical_write_bytes=128 * 448 * 1024,
                **measure,
                stats_after_write=start,
            )
        )
        # Snapshot only write deltas: verify reads are outside BPF interval.
        rows[-1]["write_requests"] = owner.stats["requests"] - start["requests"] - 2
        rows[-1]["write_submissions"] = (
            owner.stats["submissions"] - start["submissions"] - 1
        )
        s.invalidate(g)
        s.gc()
        (a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(name, measure, flush=True)
owner.close()
