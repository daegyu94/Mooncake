"""Policy transition with a prepared read pin and delayed post-CQE publication."""

import argparse
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_native import payload
from clients import ReaderSession, Writer
from coordinator import StoreError
from native_io import NativeIO
from rpc import RPC, Peer


def run(mode, library, mount, parent):
    root = parent / ("typed-fence-" + uuid.uuid4().hex)
    root.mkdir()
    peer = Peer("regions" if mode == "regions" else "objects")
    ctl, putrpc, getrpc = [RPC(peer.address) for _ in range(3)]
    io = NativeIO(library, mount, 8, 4 << 20)
    oldpath, newpath = root / "old.kv", root / "new.kv"
    oldfd, newfd = io.open(oldpath), io.open(newpath)
    old = ctl.call("create", identity="weight-A")
    w = Writer(putrpc, old, 64 << 10, 8)
    io.io(oldfd, [(0, payload(1, 0, 64 << 10))])
    w.completed(w.reserve(1), 1)
    r = ReaderSession(getrpc, old, "weight-A", mode)
    read_ready, write_ready, proceed = [threading.Event() for _ in range(3)]
    late = w.reserve(1)

    def read():
        with r.pin():
            loc = r.locations(w.region, [0])
            read_ready.set()
            assert proceed.wait(30)
            assert io.io(oldfd, loc, True) == [payload(1, 0, 64 << 10)]
        return True

    def write():
        io.io(oldfd, [(late * (64 << 10), payload(1, late, 64 << 10))])
        write_ready.set()
        assert proceed.wait(30)
        try:
            w.completed(late, 1)
        except StoreError:
            return True
        raise AssertionError("late publication accepted")

    with ThreadPoolExecutor(max_workers=2) as pool:
        read_f, write_f = pool.submit(read), pool.submit(write)
        assert read_ready.wait(30) and write_ready.wait(30)
        ctl.call("revoke", policy=old)
        assert not ctl.call("collect", policy=old)
        try:
            r.close()
        except StoreError:
            pass
        else:
            raise AssertionError("released live read pin")
        try:
            ReaderSession(ctl, old, "weight-A", mode)
        except StoreError:
            pass
        else:
            raise AssertionError("old policy reopened")
        new = ctl.call("create", identity="weight-B")
        nw = Writer(ctl, new, 64 << 10, 1)
        io.io(newfd, [(0, payload(2, 0, 64 << 10))])
        nw.completed(nw.reserve(1), 1)
        nw.drained()
        nr = ReaderSession(ctl, new, "weight-B", mode)
        with nr.pin():
            assert io.io(newfd, nr.locations(nw.region, [0]), True) == [
                payload(2, 0, 64 << 10)
            ]
        assert oldpath.exists()
        proceed.set()
        assert read_f.result(30) and write_f.result(30)
    assert not ctl.call("collect", policy=old)
    r.close()
    assert not ctl.call("collect", policy=old)
    w.drained()
    assert ctl.call("collect", policy=old)
    nr.close()
    ctl.call("revoke", policy=new)
    assert ctl.call("collect", policy=new)
    io.close_fd(oldfd)
    io.close_fd(newfd)
    io.close()
    oldpath.unlink()
    newpath.unlink()
    root.rmdir()
    for c in [ctl, putrpc, getrpc]:
        c.close()
    peer.close()
    return {
        "mode": mode,
        "old_read_valid": True,
        "new_read_valid": True,
        "late_publication_rejected": True,
        "read_pin_blocks_release": True,
        "generation_reopen_rejected": True,
        "drain_before_collect": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.output.open("x") as out:
        for repeat in range(5):
            for mode in ["cached", "regions"]:
                row = run(mode, args.library, args.mount, args.parent)
                row["repeat"] = repeat
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(repeat, flush=True)
