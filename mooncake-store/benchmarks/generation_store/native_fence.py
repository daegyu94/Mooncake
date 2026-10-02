"""Real read after invalidation remains pinned; new acquisitions are rejected."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import uuid
from generation_store import GenerationStore, NativeIO

p = argparse.ArgumentParser()
p.add_argument("--library", type=Path, required=True)
p.add_argument("--mount", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
io = NativeIO(a.library, a.mount)
s = GenerationStore(io, a.mount / "verl-lab" / ("fence-" + uuid.uuid4().hex), 2 << 20)
rows = []
for repeat in range(5):
    old = s.create()
    s.append(old, [b"a" * (448 << 10)])
    s.seal(old)
    entered = threading.Event()
    release = threading.Event()

    class Gate:
        def io(self, *args, **kwargs):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test gate")
            return io.io(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(s.get, old, [0], channel=Gate())
        assert entered.wait(timeout=10)
        s.invalidate(old)
        assert s.gc() == 0
        try:
            s.get(old, [0])
            raise AssertionError("new old-policy acquisition accepted")
        except KeyError:
            pass
        new = s.create()
        s.append(new, [b"b" * (448 << 10)])
        assert s.get(new, [0]) == [b"b" * (448 << 10)]
        release.set()
        assert future.result() == [b"a" * (448 << 10)]
        assert s.gc() == 1
        s.invalidate(new)
        s.gc()
    rows.append(
        dict(
            repeat=repeat,
            old_reader_valid=True,
            new_generation_correct=True,
            new_old_acquisition_rejected=True,
        )
    )
(a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
io.close()
print("5 native lease/fence checks passed")
