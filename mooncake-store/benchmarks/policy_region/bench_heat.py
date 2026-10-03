"""Causal trace replay on a pinned residency owner and real native3FS IO.

The stronger heat_lru/adaptive pair shares admission and cost-ranked victims.
Original lru/drain are diagnostic controls. Traces are defined independently
of the policy; no workload label or future event enters the cache algorithm.
"""

import argparse
import hashlib
import json
import resource
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_resident import payload
from heat_store import HeatResidentStore
from native_io import NativeIO
from resident_store import ResidentStore


def trace(store, ios, size, pattern, epochs=4, concurrency=1):
    phases, hits, snapshots = [], [], []
    verified = 0
    sizes = {}

    def create(epoch):
        stride = size * ([1, 4, 7, 16, 64][epoch % 5] if pattern == "mixed_size" else 1)
        policy = store.create(f"weights-{epoch}", stride)
        sizes[policy] = stride
        return policy

    def measure(label, fn):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        result = fn()
        phases.append(
            {
                "label": label,
                "wall_ns": time.perf_counter_ns() - wall,
                "cpu_ns": time.process_time_ns() - cpu,
            }
        )
        return result

    def put(policy, begin, end):
        for ordinal in range(begin, end):
            assert (
                store.put(policy, payload(policy, ordinal, sizes[policy]))[0] == ordinal
            )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:

        def get(policy, lease, keys, label, passes=1):
            nonlocal verified
            before = store.snapshot()["metrics"]

            def work(index):
                subset = keys[index :: len(ios)]
                if not subset:
                    return 0
                with store.read(lease, subset, ios[index]) as values:
                    assert values == [payload(policy, k, sizes[policy]) for k in subset]
                return len(subset)

            for _ in range(passes):
                verified += sum(
                    measure(label, lambda: list(pool.map(work, range(len(ios)))))
                )
            after = store.snapshot()["metrics"]
            hits.append(
                {
                    "label": label,
                    "requested": len(keys) * passes * sizes[policy],
                    "ram_bytes": after["ram_read_bytes"] - before["ram_read_bytes"],
                    "dfs_bytes": after["dfs_read_bytes"] - before["dfs_read_bytes"],
                }
            )

        old = create(0)
        old_lease = store.acquire(old, "weights-0")
        measure("initial_put", lambda: put(old, 0, 16))
        for epoch in range(1, epochs):
            measure("revoke", lambda old=old: store.revoke(old))
            new = create(epoch)
            lease = store.acquire(new, f"weights-{epoch}")
            delay = (
                0
                if pattern == "barrier"
                else 16
                if pattern in ["late", "working_set"]
                else 8
            )
            measure("put_before_drain", lambda new=new, delay=delay: put(new, 0, delay))
            hot = pattern == "hot" or (
                pattern == "cold_to_hot" and epoch >= epochs // 2
            )
            hot = hot or (pattern == "hot_to_cold" and epoch < epochs // 2)
            if hot:
                get(new, lease, list(range(min(delay, 8))), "active_hot", 8)
            elif pattern == "working_set":
                get(new, lease, list(range(4)), "active_working_set", 8)
            elif pattern == "scan":
                get(new, lease, list(range(min(delay, 8))), "active_scan")
            get(old, old_lease, list(range(16)), "old_read")
            snapshots.append(store.snapshot())
            measure("release", lambda old_lease=old_lease: store.release(old_lease))
            measure("put_after_drain", lambda new=new, delay=delay: put(new, delay, 16))
            old, old_lease = new, lease
        get(old, old_lease, list(range(16)), "last_read")
        measure("last_revoke", lambda: store.revoke(old))
        measure("last_release", lambda: store.release(old_lease))
    final = store.snapshot()
    assert final["resident_bytes"] == final["policies"] == final["fds"] == 0
    assert final["metrics"]["peak_resident_bytes"] <= store.budget
    return {
        "phases": phases,
        "hits": hits,
        "checkpoints": snapshots,
        "verified": verified,
        "logical_get_bytes": sum(h["requested"] for h in hits),
        "object_sizes": sizes,
        "final": final,
    }


def run(
    mode,
    size,
    pattern,
    library,
    mount,
    parent,
    epochs=4,
    concurrency=1,
    budget=None,
    io_factory=None,
):
    root = parent / ("heat-" + uuid.uuid4().hex)
    root.mkdir()
    ios = [(io_factory or NativeIO)(library, mount, 8, 4 << 20) for _ in range(8)]
    budget = 8 * size if budget is None else budget
    owner = (
        HeatResidentStore(ios[0], root, budget, mode)
        if mode in ["heat_lru", "adaptive"]
        else ResidentStore(ios[0], root, budget, mode)
    )
    try:
        row = trace(owner, ios, size, pattern, epochs, concurrency)
        stats = {key: sum(io.stats[key] for io in ios) for key in ios[0].stats}
        metrics = row["final"]["metrics"]
        assert stats["read_bytes"] == metrics["dfs_read_bytes"]
        assert (
            stats["write_bytes"] + metrics["discard_dirty_bytes"]
            == metrics["accepted_bytes"]
        )
        assert (
            row["logical_get_bytes"] == stats["read_bytes"] + metrics["ram_read_bytes"]
        )
        assert stats["fd_register"] == stats["fd_deregister"]
        row.update(
            mode=mode,
            size=size,
            pattern=pattern,
            epochs=epochs,
            concurrency=concurrency,
            budget=budget,
            native=stats,
            suite_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            loaded_libraries=[
                line.strip()
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "usrbio" in line or "hf3fs_api" in line
            ],
        )
        return row
    finally:
        for io in ios:
            io.close()
        if not list(root.iterdir()):
            root.rmdir()


def main():
    parser = argparse.ArgumentParser()
    for key in ["library", "mount", "parent", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["lru", "drain", "heat_lru", "adaptive"], required=True
    )
    parser.add_argument(
        "--pattern",
        choices=[
            "cold",
            "hot",
            "cold_to_hot",
            "hot_to_cold",
            "scan",
            "late",
            "barrier",
            "working_set",
            "mixed_size",
        ],
        default="cold",
    )
    parser.add_argument("--size", type=int, default=448 << 10)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if not args.mount.is_mount() or not args.parent.is_dir() or args.epochs < 2:
        raise ValueError("native mount/parent and >=2 epochs required")
    row = run(
        args.mode,
        args.size,
        args.pattern,
        args.library,
        args.mount,
        args.parent,
        args.epochs,
        args.concurrency,
        args.budget,
    )
    row["bridge_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
    with args.output.open("x") as output:
        output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
