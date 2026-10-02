"""Confirm range reduction at storage-engine AIO completion bytes, not only client counters."""

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
StorageIO = importlib.import_module("storage_aio_probe").StorageIO

io = NativeIO(a.library, a.mount, depth=8)
s = GenerationStore(io, a.mount / "verl-lab" / ("range-" + uuid.uuid4().hex), 1 << 30)
g = s.create()
values = [bytes([i]) * (448 << 10) for i in range(128)]
s.append(g, values)
s.seal(g)
keys = list(range(0, 128, 16))
rows = []
for repeat in range(5):
    for whole in [16, 0] if repeat % 2 == 0 else [0, 16]:
        name = f"r{repeat}-" + ("whole" if whole else "range")
        out = a.output / name
        out.mkdir()
        before = dict(io.stats)
        with StorageIO(out):
            result, m = timed(lambda: s.get(g, keys, whole_extent=whole))
        assert result == [values[k] for k in keys]
        rows.append(
            dict(
                name=name,
                repeat=repeat,
                whole=whole,
                requested_bytes=len(keys) * (448 << 10),
                read_bytes=io.stats["read_bytes"] - before["read_bytes"],
                **m,
            )
        )
        (a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
s.invalidate(g)
s.gc()
io.close()
print("Range physical IO A/B complete")
