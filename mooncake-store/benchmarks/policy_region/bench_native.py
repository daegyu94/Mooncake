"""Experimental control plane plus real native 3FS IO, not Mooncake master RPC."""

import argparse
import json
import os
import resource
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_control import timed
from clients import ReaderSession, Writer
from native_io import NativeIO
from rpc import RPC, Peer


def payload(epoch, ordinal, size):
    unit = epoch.to_bytes(8, "little") + ordinal.to_bytes(8, "little")
    return unit * (size // 16)


def run(
    mode,
    size,
    concurrency,
    pattern,
    library,
    mount,
    parent,
    epochs=1,
    overlap=1,
    count=32,
):
    root = parent / ("typed-region-" + uuid.uuid4().hex)
    root.mkdir()
    peer = Peer("regions" if mode == "regions" else "objects")
    connections = [RPC(peer.address) for _ in range(9)]
    ctl = connections[0]
    ios = [NativeIO(library, mount, 8, 4 << 20) for _ in range(8)]
    live, timings, peaks = [], [], []
    verified = 0
    initial = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter_ns()

    def retire(item):
        p, readers, fd, path = item
        for r in readers:
            r.close()
        assert ctl.call("collect", policy=p)
        ios[0].close_fd(fd)
        path.unlink()

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for epoch in range(epochs):
                identity = f"weight-{epoch}/model-A/layout-v1"
                p = ctl.call("create", identity=identity)
                w = Writer(ctl, p, size, count)
                path = root / f"{ctl.boot}-{p}-{w.region}.kv"
                fd = ios[0].open(path)
                readers = [ReaderSession(c, p, identity, mode) for c in connections[1:]]

                def put(w=w, fd=fd, epoch=epoch):
                    for first in range(0, count, 8):
                        n = min(8, count - first)
                        slot = w.reserve(n)
                        ios[0].io(
                            fd,
                            [
                                (o * size, payload(epoch, o, size))
                                for o in range(slot, slot + n)
                            ],
                        )
                        w.completed(slot, n)
                    w.drained()

                _, put_time = timed(put)
                keys = list(range(0, count, 1 if pattern == "dense" else 8))
                tasks = [keys[i::8] for i in range(8)]

                def read(index, readers=readers, tasks=tasks, w=w, fd=fd, epoch=epoch):
                    r = readers[index]
                    with r.pin():
                        locations = r.locations(w.region, tasks[index])
                        data = ios[index].io(fd, locations, read=True)
                        assert data == [payload(epoch, o, size) for o in tasks[index]]
                    return len(data)

                reads = []
                for _ in range(2):
                    checked, elapsed = timed(lambda: list(pool.map(read, range(8))))
                    verified += sum(checked)
                    reads.append(elapsed)
                _, revoke = timed(lambda p=p: ctl.call("revoke", policy=p))
                live.append((p, readers, fd, path))
                state = ctl.call("stats")
                peaks.append(
                    {
                        "policies": state["policies"],
                        "regions": state["regions"],
                        "metadata_bytes": state["metadata_bytes"],
                        "descriptor_bytes": sum(
                            r.bytes() for _, rs, _, _ in live for r in rs
                        ),
                    }
                )
                gc = {"wall_ns": 0, "cpu_ns": 0}
                if len(live) >= overlap:
                    _, gc = timed(lambda: retire(live.pop(0)))
                timings.append(
                    {"put": put_time, "reads": reads, "revoke": revoke, "gc": gc}
                )
            while live:
                _, gc = timed(lambda: retire(live.pop(0)))
                timings[-1]["gc"] = {k: timings[-1]["gc"][k] + gc[k] for k in gc}
        final = ctl.call("stats")
        assert final["policies"] == final["regions"] == final["objects"] == 0
        assert final["transport"]["rpcs"] == sum(c.stats["rpcs"] for c in connections)
        io = {key: sum(i.stats[key] for i in ios) for key in ios[0].stats}
        assert io["read_bytes"] == verified * size
        assert io["write_bytes"] == epochs * count * size
        assert io["fd_register"] == io["fd_deregister"] == epochs
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "mode": mode,
            "size": size,
            "concurrency": concurrency,
            "pattern": pattern,
            "epochs": epochs,
            "overlap": overlap,
            "count": count,
            "timings": timings,
            "server": final,
            "io": io,
            "peaks": peaks,
            "verified": verified,
            "instrumented_run_wall_ns": time.perf_counter_ns() - started,
            "voluntary_switches": usage.ru_nvcsw - initial.ru_nvcsw,
            "involuntary_switches": usage.ru_nivcsw - initial.ru_nivcsw,
            "client_peak_rss_kib": usage.ru_maxrss,
            "region_tail_slack_bytes": 0,
            "live_data_gc_copy_bytes": 0,
            "native_library": str(library.resolve()),
            "loaded_libraries": [
                s.strip()
                for s in Path("/proc/self/maps").read_text().splitlines()
                if ".so" in s and ("usrbio" in s or "hf3fs" in s)
            ],
            "leftover_files": len(list(root.iterdir())),
        }
    finally:
        # No recursive cleanup: an error leaves only this UUID's data quarantined.
        for io in ios:
            io.close()
        for c in connections:
            c.close()
        peer.close()
        if not list(root.iterdir()):
            root.rmdir()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--suite", choices=["matrix", "long", "smoke"], default="matrix")
    args = p.parse_args()
    if not args.parent.is_dir() or not os.path.ismount(args.mount):
        raise RuntimeError("existing dedicated 3FS parent/mount required")
    if args.suite == "matrix":
        cases = [
            (size, c, pat, 1, 1, 32)
            for size in [64 << 10, 448 << 10, 1 << 20, 4 << 20]
            for c in [1, 4, 8]
            for pat in ["dense", "sparse"]
        ]
    elif args.suite == "long":
        cases = [(64 << 10, 4, "dense", 128, overlap, 8) for overlap in [1, 4]]
    else:
        cases = [(448 << 10, 1, "dense", 1, 1, 32)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for repeat in range(args.repeat):
            for case in cases:
                for mode in (
                    ["cached", "regions"] if repeat % 2 == 0 else ["regions", "cached"]
                ):
                    size, c, pat, epochs, overlap, count = case
                    row = run(
                        mode,
                        size,
                        c,
                        pat,
                        args.library,
                        args.mount,
                        args.parent,
                        epochs,
                        overlap,
                        count,
                    )
                    row["repeat"] = repeat
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                print(repeat, case, flush=True)


if __name__ == "__main__":
    main()
