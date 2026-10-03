"""Native data with controlled post-CQE publication/migration gates."""

import argparse
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coordinator import StoreError
from native_io import NativeIO
from resident_store import ResidentStore


class CompletionGate:
    def __init__(self, native):
        self.native = native
        self.enter, self.resume = threading.Event(), threading.Event()

    def open(self, path):
        return self.native.open(path)

    def close_fd(self, fd):
        return self.native.close_fd(fd)

    def io(self, fd, ranges, read=False):
        result = self.native.io(fd, ranges, read)
        if not read:
            self.enter.set()
            if not self.resume.wait(30):
                raise TimeoutError("post-completion gate")
        return result


def run(kind, library, mount, parent):
    root = parent / ("residency-fence-" + uuid.uuid4().hex)
    root.mkdir()
    native = NativeIO(library, mount, 8, 4 << 20)
    gate = CompletionGate(native)
    store = ResidentStore(
        gate, root, 64 << 10, "eager" if kind == "late-put" else "drain"
    )
    a = store.create("weight-A", 64 << 10)
    if kind == "late-put":
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(store.put, a, b"A" * (64 << 10))
            assert gate.enter.wait(30)
            assert not store.revoke(a)
            gate.resume.set()
            try:
                future.result(30)
            except StoreError:
                pass
            else:
                raise AssertionError("late publication accepted")
        assert not store.policies and not list(root.iterdir())
    else:
        lease = store.acquire(a, "weight-A")
        store.put(a, b"A" * (64 << 10))
        b = store.create("weight-B", 64 << 10)
        new_lease = store.acquire(b, "weight-B")
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(store.put, b, b"B" * (64 << 10))
            assert gate.enter.wait(30)
            assert not store.revoke(a)
            with store.read(lease, [0]) as values:
                assert values[0] == b"A" * (64 << 10)
                gate.resume.set()
                try:
                    future.result(30)
                except BufferError:
                    pass
                else:
                    raise AssertionError("borrowed source freed")
                assert values[0] == b"A" * (64 << 10)
                try:
                    store.acquire(a, "weight-A")
                except StoreError:
                    pass
                else:
                    raise AssertionError("retired namespace reopened")
            assert store.release(lease)
        store.put(b, b"B" * (64 << 10))
        with store.read(new_lease, [0]) as values:
            assert values[0] == b"B" * (64 << 10)
        store.revoke(b)
        assert store.release(new_lease)
    result = {
        "kind": kind,
        "correct": True,
        "final": store.snapshot(),
        "native": native.stats.copy(),
    }
    native.close()
    assert not list(root.iterdir())
    root.rmdir()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["library", "mount", "parent", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    with args.output.open("x") as output:
        for repeat in range(5):
            for kind in ["migrate", "late-put"]:
                row = run(kind, args.library, args.mount, args.parent)
                row["repeat"] = repeat
                output.write(json.dumps(row) + "\n")
                output.flush()
            print(repeat, flush=True)
