"""A/B segmented/vector I/O versus L01 mutable tail + the same adjacent plan.

--backend stub is explicitly a POSIX/HF3FS ABI-copy microbenchmark. Only native
runs with the real library and mount qualify as native3FS component validation.
Neither executes production Mooncake Master, GPU transfers or a veRL trainer.
"""

import argparse
import hashlib
import importlib
import json
import math
import resource
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from generation_store import NativeIO
from stream_store import PlannedTailStore, VectorTailStore
from vector_io import VectorIO


def value_for(epoch, ordinal, size):
    header = epoch.to_bytes(8, "little") + ordinal.to_bytes(8, "little")
    return (header * math.ceil(size / len(header)))[:size]


def read_groups(count, pattern):
    width = math.ceil(count / 8)
    groups = []
    for begin in range(0, count, width):
        keys = list(range(begin, min(count, begin + width)))
        if pattern == "sparse":
            keys = keys[::4]
        elif pattern == "duplicates":
            keys = list(reversed(keys)) + keys[:1]
        elif pattern == "mixed":
            keys = keys[::2] + list(reversed(keys[1::2]))
        groups.append(keys)
    return groups


def run_case(args, size, pattern, concurrency, variant, epochs):
    io_cls = NativeIO if variant == "mutable-adjacent" else VectorIO
    library = (
        args.baseline_library if variant == "mutable-adjacent" else args.vector_library
    )
    channels = [
        io_cls(library, args.mount, args.depth, args.slot_bytes) for _ in range(9)
    ]
    root = args.parent / ("vector-stream-" + uuid.uuid4().hex)
    cls = PlannedTailStore if variant == "mutable-adjacent" else VectorTailStore
    store = cls(channels[0], root, 1 << 34, tail_budget=args.tail_budget)
    groups = read_groups(args.objects, pattern)
    phases = {
        label: {"wall_ns": 0, "cpu_ns": 0}
        for label in ("append", "read", "flush", "transition", "gc")
    }
    logical_get = verified = 0
    request_ns, events, live = [], [], []
    pool = ThreadPoolExecutor(max_workers=concurrency)

    def measure(label, function):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        result = function()
        phases[label]["wall_ns"] += time.perf_counter_ns() - wall
        phases[label]["cpu_ns"] += time.process_time_ns() - cpu
        return result

    def read_group(gid, epoch, index):
        started = time.perf_counter_ns()
        result = store.get(gid, groups[index], channels[index + 1])
        assert result == [value_for(epoch, ordinal, size) for ordinal in groups[index]]
        return time.perf_counter_ns() - started

    def retire(gid, lease, epoch):
        nonlocal logical_get, verified
        measure("transition", lambda: store.invalidate(gid))
        # Retirement revokes new lookup, but this already-issued lease remains
        # valid through physical completion, including a RAM/DFS split object.
        tail_key = args.objects - 1
        result = measure(
            "read", lambda: store.read_lease(lease, [tail_key], channels[1])
        )
        assert result == [value_for(epoch, tail_key, size)]
        logical_get += size
        verified += 1
        measure("transition", lambda: store.release(lease))
        try:
            store.get(gid, [0])
        except KeyError:
            pass
        else:
            raise AssertionError("retired generation remained discoverable")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    started, cpu = time.perf_counter_ns(), time.process_time_ns()
    try:
        for epoch in range(epochs):
            gid = store.create()
            lease = store.acquire(gid)
            # Both variants receive identical immutable bytes. Their ownership
            # contract differs in how retained immutable input is represented.
            values = [
                value_for(epoch, ordinal, size) for ordinal in range(args.objects)
            ]
            for payload in values:
                measure(
                    "append",
                    lambda payload=payload, gid=gid: store.append(gid, [payload]),
                )
            store.seal(gid)
            for _ in range(args.read_passes):
                latencies = measure(
                    "read",
                    lambda gid=gid, epoch=epoch: list(
                        pool.map(
                            lambda i: read_group(gid, epoch, i), range(len(groups))
                        )
                    ),
                )
                request_ns.extend(latencies)
                requested = sum(map(len, groups))
                logical_get += requested * size
                verified += requested
            if args.flush == "complete":
                measure("flush", lambda gid=gid: store.flush(gid))
            live.append((gid, lease, epoch))
            if len(live) > args.overlap:
                retire(*live.pop(0))
            if epoch % 8 == 7:
                measure("gc", store.gc)
            events.append(
                {
                    "epoch": epoch,
                    "active": len(store.generations),
                    "retired": len(store.retired),
                    "logical_tail_bytes": store.resident_tail_bytes,
                    "fds": len(list(Path("/proc/self/fd").iterdir())),
                }
            )
        for item in live:
            retire(*item)
        measure("gc", store.gc)
        elapsed, consumed = (
            time.perf_counter_ns() - started,
            time.process_time_ns() - cpu,
        )
        after = resource.getrusage(resource.RUSAGE_SELF)
        native = {
            key: sum(channel.stats[key] for channel in channels)
            for key in channels[0].stats
        }
        accepted = epochs * args.objects * size
        assert native["write_bytes"] + store.metrics["discarded_tail_bytes"] == accepted
        assert store.metrics["accepted_bytes"] == accepted
        assert native["read_bytes"] == store.metrics["unique_disk_requested_bytes"]
        assert store.metrics["resident_peak_bytes"] <= args.tail_budget
        assert (
            not store.generations
            and not store.retired
            and store.resident_tail_bytes == 0
        )
        assert not list(root.iterdir())
        # Narrow, explicit copies actually executed in adapter/store. Ordinary
        # allocator zero-fill, libc memmove and kernel/server copies are absent.
        if variant == "mutable-adjacent":
            copy_components = {
                "tail_append_snapshot": store.metrics["explicit_tail_copy_bytes"],
                "ctypes_write_staging": native["write_bytes"],
                "bridge_registered_copy": native["copy_bytes"],
                "ctypes_read_output": native["read_bytes"],
                "ram_tail_snapshot": store.metrics.get(
                    "read_tail_snapshot_copy_bytes", 0
                ),
                "python_scatter": store.metrics.get("read_scatter_copy_bytes", 0),
            }
        else:
            copy_components = {
                "mutable_input_ownership": store.metrics["ownership_copy_bytes"],
                "registered_gather": native["gather_copy_bytes"],
                "registered_scatter": native["scatter_copy_bytes"],
                "ram_suffix_and_output": native["ram_copy_bytes"],
            }
        return {
            "loaded_libraries": [
                line.strip()
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "vector_bridge" in line
                or "usrbio_bridge" in line
                or "hf3fs_api" in line
            ],
            "variant": variant,
            "size": size,
            "pattern": pattern,
            "concurrency": concurrency,
            "epochs": epochs,
            "objects": args.objects,
            "overlap": args.overlap,
            "tail_budget": args.tail_budget,
            "depth": args.depth,
            "slot_bytes": args.slot_bytes,
            "read_passes": args.read_passes,
            "flush": args.flush,
            "wall_ns": elapsed,
            "cpu_ns": consumed,
            "logical_put_bytes": accepted,
            "logical_get_bytes": logical_get,
            "verified_gets": verified,
            "io": native,
            "metrics": store.metrics,
            "phases": phases,
            "request_ns": request_ns,
            "events": events,
            "explicit_copy_components": copy_components,
            "explicit_copy_bytes": sum(copy_components.values()),
            "voluntary_switches": after.ru_nvcsw - usage.ru_nvcsw,
            "involuntary_switches": after.ru_nivcsw - usage.ru_nivcsw,
            "process_high_water_rss_kib": after.ru_maxrss,
        }
    finally:
        pool.shutdown(wait=True)
        for channel in channels:
            channel.close()
        if root.exists() and not list(root.iterdir()):
            root.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-library", "vector-library", "mount", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "stub"), required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64 << 10, 256 << 10, 448 << 10, 1 << 20, 4 << 20],
    )
    parser.add_argument(
        "--patterns",
        choices=("dense", "sparse", "duplicates", "mixed"),
        nargs="+",
        default=["dense", "sparse"],
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--objects", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--read-passes", type=int, default=2)
    parser.add_argument("--tail-budget", type=int, default=4 << 20)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--slot-bytes", type=int, default=4 << 20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--flush", choices=("complete", "expire"), default="complete")
    parser.add_argument("--probe-module", type=Path)
    args = parser.parse_args()
    if args.backend == "native" and not args.mount.is_mount():
        raise ValueError("native requires an actual 3FS mount")
    if (
        not args.parent.is_dir()
        or args.objects < 1
        or args.epochs < 1
        or args.overlap < 1
        or args.read_passes < 1
        or any(size <= 0 for size in args.sizes)
        or any(c < 1 or c > 8 for c in args.concurrency)
    ):
        raise ValueError("invalid workload or parent namespace")
    args.output.mkdir(parents=True, exist_ok=False)
    probe = None
    if args.probe_module:
        if args.backend != "native":
            raise ValueError("3FS server probe cannot label a POSIX stub run")
        sys.path.insert(0, str(args.probe_module))
        probe = importlib.import_module("storage_aio_probe").StorageIO
    manifest = {
        "backend": args.backend,
        "validation": "Native3FS independent component"
        if args.backend == "native"
        else "POSIX/HF3FS ABI stub microbenchmark; not native3FS",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip(),
        "sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.*")
            if path.suffix in (".py", ".c")
        },
        "baseline_bridge_sha256": hashlib.sha256(
            args.baseline_library.read_bytes()
        ).hexdigest(),
        "vector_bridge_sha256": hashlib.sha256(
            args.vector_library.read_bytes()
        ).hexdigest(),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "registered_channels": 9,
        "registered_bytes": 9 * args.depth * args.slot_bytes,
        "copy_scope": "explicit user-space source copies only; excludes zero-fill, tail memmove, kernel/server copies, GPU D2H",
        "cache_state": "unique per-run namespace; service/SSD cache not flushed; A/B order rotated",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "rows.jsonl").open("x") as output:
        for repeat in range(args.repeat):
            sizes = (
                args.sizes[repeat % len(args.sizes) :]
                + args.sizes[: repeat % len(args.sizes)]
            )
            for size in sizes:
                for pattern in args.patterns:
                    for concurrency in args.concurrency:
                        pair = {}
                        variants = ["mutable-adjacent", "vector-adjacent"]
                        if repeat % 2:
                            variants.reverse()
                        for variant in variants:
                            name = f"s{size}-{pattern}-c{concurrency}-e{args.epochs}-r{repeat}-{variant}"
                            directory = args.output / name
                            directory.mkdir()
                            with probe(directory) if probe else nullcontext():
                                row = run_case(
                                    args,
                                    size,
                                    pattern,
                                    concurrency,
                                    variant,
                                    args.epochs,
                                )
                            row.update(name=name, repeat=repeat)
                            pair[variant] = row
                            output.write(json.dumps(row) + "\n")
                            output.flush()
                            print(
                                name,
                                f"cpu_ms={row['cpu_ns'] / 1e6:.3f} copy={row['explicit_copy_bytes']} write={row['io']['write_bytes']} read={row['io']['read_bytes']}",
                                flush=True,
                            )
                        a, b = pair["mutable-adjacent"], pair["vector-adjacent"]
                        for metric in (
                            "logical_put_bytes",
                            "logical_get_bytes",
                            "verified_gets",
                        ):
                            assert a[metric] == b[metric]
                        for metric in (
                            "write_bytes",
                            "read_bytes",
                            "requests",
                            "submissions",
                            "fd_register",
                            "fd_deregister",
                        ):
                            assert a["io"][metric] == b["io"][metric]
    manifest["completed_unix_ns"] = time.time_ns()
    manifest["loaded_libraries"] = [
        line.strip()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if ".so" in line
        and any(
            part in line for part in ("usrbio", "hf3fs", "vector_bridge", "vector_stub")
        )
    ]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
