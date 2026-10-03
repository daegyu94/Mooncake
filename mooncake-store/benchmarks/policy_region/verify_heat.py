"""Native post-GET/pre-admission revoke and policy identity fencing."""

import argparse
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from coordinator import StoreError
from heat_store import HeatResidentStore
from native_io import NativeIO


class ReadGate:
    def __init__(self, native):
        self.native = native
        self.enter, self.resume = threading.Event(), threading.Event()
        self.enabled = False

    def open(self, path):
        return self.native.open(path)

    def close_fd(self, fd):
        self.native.close_fd(fd)

    def io(self, fd, ranges, read=False):
        result = self.native.io(fd, ranges, read)
        if read and self.enabled:
            self.enter.set()
            if not self.resume.wait(30):
                raise TimeoutError("post-GET completion gate")
        return result


def run(library, mount, parent, mode):
    root = parent / ("heat-fence-" + uuid.uuid4().hex)
    root.mkdir()
    io = NativeIO(library, mount, 8, 4 << 20)
    gate = ReadGate(io)
    store = HeatResidentStore(gate, root, 64 << 10, mode)
    a = store.create("weight-A", 64 << 10)
    lease = store.acquire(a, "weight-A")
    store.put(a, b"A" * (64 << 10))
    store.put(a, b"B" * (64 << 10))
    with store.read(lease, [0]) as values:
        assert values[0] == b"A" * (64 << 10)
    gate.enabled = True

    def get():
        with store.read(lease, [0]) as values:
            assert values[0] == b"A" * (64 << 10)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(get)
        assert gate.enter.wait(30)
        assert not store.revoke(a)
        try:
            store.release(lease)
        except StoreError:
            pass
        else:
            raise AssertionError("release accepted while GET owns completion")
        gate.resume.set()
        future.result(30)
    assert store.policies[a].blocks[0].data is None
    assert store.metrics["admission_stale_rejections"] == 1
    for invalid in [replace(lease, identity="weight-B"), replace(lease, boot="wrong")]:
        try:
            with store.read(invalid, [0]):
                pass
        except StoreError:
            pass
        else:
            raise AssertionError("stale identity accepted")
    assert store.release(lease)
    b = store.create("weight-B", 64 << 10)
    new = store.acquire(b, "weight-B")
    store.put(b, b"N" * (64 << 10))
    with store.read(new, [0]) as values:
        assert values[0] == b"N" * (64 << 10)
    store.revoke(b)
    store.release(new)
    result = {
        "mode": mode,
        "correct": True,
        "native": io.stats.copy(),
        "final": store.snapshot(),
    }
    io.close()
    assert not list(root.iterdir())
    root.rmdir()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ["library", "mount", "parent", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as output:
        for repeat in range(5):
            for mode in ["heat_lru", "adaptive"]:
                row = run(args.library, args.mount, args.parent, mode)
                row["repeat"] = repeat
                output.write(json.dumps(row) + "\n")
                output.flush()
