"""Five real 3FS reads pinned across invalidation of a mixed RAM/DFS object."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import uuid

from generation_store import NativeIO
from tail_store import TailBufferedStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    io = NativeIO(a.library, a.mount, depth=8)
    root = a.mount / "verl-lab" / ("tail-fence-" + uuid.uuid4().hex)
    s = TailBufferedStore(io, root, 32 << 20)
    rows = []
    for repeat in range(5):
        old = s.create()
        value = b"a" * (448 << 10)
        s.append(old, [value] * 3)
        s.seal(old)
        assert s.generations[old].persisted == 1 << 20
        entered, proceed = threading.Event(), threading.Event()

        class Gate:
            def io(self, *args, **kwargs):
                entered.set()
                if not proceed.wait(10):
                    raise TimeoutError("test gate")
                return io.io(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(s.get, old, [2, 0, 2], channel=Gate())
            try:
                assert entered.wait(10)
                s.invalidate(old)
                assert s.gc() == 0
                assert len(s.retired[old].tail) == 320 << 10
                try:
                    s.get(old, [0])
                    raise AssertionError("new old-policy acquisition accepted")
                except KeyError:
                    pass
                try:
                    s.append(old, [b"late"])
                    raise AssertionError("late PUT accepted")
                except KeyError:
                    pass
                new = s.create()
                s.append(new, [b"b" * (448 << 10)] * 3)
                assert s.get(new, [0, 2]) == [b"b" * (448 << 10)] * 2
            finally:
                proceed.set()
            assert future.result() == [value] * 3
            assert not s.retired[old].tail
            assert s.gc() == 1
            s.invalidate(new)
            assert s.gc() == 1
        rows.append(
            dict(
                repeat=repeat,
                mixed_ram_dfs_read_correct=True,
                old_lease_preserved=True,
                new_old_acquisition_rejected=True,
                late_put_rejected=True,
                new_policy_correct=True,
            )
        )
    assert s.resident_tail_bytes == 0
    (a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    io.close()
    print("5 native mixed RAM/DFS lease and fencing checks passed")


if __name__ == "__main__":
    main()
