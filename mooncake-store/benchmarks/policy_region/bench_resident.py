"""Native3FS comparison: eager diagnostic vs ordinary LRU vs drain protection."""

import argparse
import json
import resource
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from native_io import NativeIO
from resident_store import ResidentStore


def payload(policy, ordinal, size):
    return (policy.to_bytes(8, "little") + ordinal.to_bytes(8, "little")) * (size // 16)


def run(
    mode,
    size,
    slots,
    release_after,
    pattern,
    concurrency,
    library,
    mount,
    parent,
    epochs=1,
):
    root = parent / ("residency-" + uuid.uuid4().hex)
    root.mkdir()
    ios = [NativeIO(library, mount, 8, 4 << 20) for _ in range(8)]
    budget = int(slots * size)
    store = ResidentStore(ios[0], root, budget, mode)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    phase = []
    hits = []
    checkpoints = []
    verified = 0

    def measure(label, fn):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        value = fn()
        phase.append(
            {
                "label": label,
                "wall_ns": time.perf_counter_ns() - wall,
                "cpu_ns": time.process_time_ns() - cpu,
            }
        )
        return value

    def put(p, start, end):
        for k in range(start, end):
            ordinal, receipt = store.put(p, payload(p, k, size))
            assert ordinal == k and receipt in ["resident-only", "dfs-complete"]

    def get(p, lease, keys, label, passes=1):
        nonlocal verified
        before = store.snapshot()["metrics"]

        def work(index):
            subset = keys[index::8]
            if not subset:
                return 0
            with store.read(lease, subset, ios[index]) as values:
                assert values == [payload(p, o, size) for o in subset]
            return len(subset)

        for _ in range(passes):
            verified += sum(measure(label, lambda: list(pool.map(work, range(8)))))
        after = store.snapshot()["metrics"]
        hits.append(
            {
                "label": label,
                "requested": len(keys) * passes * size,
                "ram_bytes": after["ram_read_bytes"] - before["ram_read_bytes"],
                "dfs_bytes": after["dfs_read_bytes"] - before["dfs_read_bytes"],
                "active_workers": min(concurrency, min(len(keys), 8)),
            }
        )

    def drop(p, lease=None):
        store.revoke(p)
        if lease is not None:
            store.release(lease)

    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        if epochs == 1:
            old = store.create("weights-old", size)
            old_l = store.acquire(old, "weights-old")
            measure("old_put", lambda: put(old, 0, 8))
            measure("old_revoke", lambda: store.revoke(old))
            new = store.create("weights-new", size)
            new_l = store.acquire(new, "weights-new")
            measure("new_put_before_drain", lambda: put(new, 0, release_after))
            if pattern == "active-hot":
                get(new, new_l, list(range(min(8, release_after))), "active_hot", 8)
            get(
                old,
                old_l,
                list(range(0, 8, 4 if pattern == "sparse" else 1)),
                "old_read",
            )
            checkpoints.append(store.snapshot())
            measure("old_release", lambda: store.release(old_l))
            measure("new_put_after_drain", lambda: put(new, release_after, 16))
            get(
                new,
                new_l,
                list(range(0, 16, 4 if pattern == "sparse" else 1)),
                "new_read",
            )
            measure("new_retire", lambda: drop(new, new_l))
        else:
            # A real overlapping old read lease and new writer at every step.
            old = store.create("weights-0", size)
            old_l = store.acquire(old, "weights-0")
            measure("initial_put", lambda: put(old, 0, 16))
            for epoch in range(1, epochs):
                measure("revoke", lambda old=old: store.revoke(old))
                new = store.create(f"weights-{epoch}", size)
                new_l = store.acquire(new, f"weights-{epoch}")
                measure("put_before_drain", lambda new=new: put(new, 0, release_after))
                get(old, old_l, list(range(16)), "old_read")
                measure("release", lambda old_l=old_l: store.release(old_l))
                measure("put_after_drain", lambda new=new: put(new, release_after, 16))
                checkpoints.append(store.snapshot())
                old, old_l = new, new_l
            get(old, old_l, list(range(16)), "last_read")
            measure("last_retire", lambda: drop(old, old_l))
        final = store.snapshot()
        stats = {k: sum(io.stats[k] for io in ios) for k in ios[0].stats}
        assert final["resident_bytes"] == final["policies"] == final["fds"] == 0
        assert not list(root.iterdir())
        assert (
            final["metrics"]["accepted_bytes"]
            == stats["write_bytes"] + final["metrics"]["discard_dirty_bytes"]
        )
        assert final["metrics"]["peak_resident_bytes"] <= budget
        assert (
            verified * size
            == final["metrics"]["ram_read_bytes"] + final["metrics"]["dfs_read_bytes"]
        )
        assert stats["read_bytes"] == final["metrics"]["dfs_read_bytes"]
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "mode": mode,
            "size": size,
            "slots": slots,
            "budget": budget,
            "release_after": release_after,
            "pattern": pattern,
            "concurrency": concurrency,
            "epochs": epochs,
            "phases": phase,
            "hits": hits,
            "checkpoints": checkpoints,
            "final": final,
            "native": stats,
            "verified": verified,
            "voluntary_switches": usage.ru_nvcsw - usage_before.ru_nvcsw,
            "involuntary_switches": usage.ru_nivcsw - usage_before.ru_nivcsw,
            "suite_peak_rss_kib": usage.ru_maxrss,
            "loaded_libraries": [
                s.strip()
                for s in Path("/proc/self/maps").read_text().splitlines()
                if ".so" in s and ("hf3fs" in s or "usrbio" in s)
            ],
        }
    finally:
        pool.shutdown(wait=True)
        for io in ios:
            io.close()
        # A failure intentionally leaves this UUID namespace for diagnosis.
        if not list(root.iterdir()):
            root.rmdir()


def main():
    p = argparse.ArgumentParser()
    for name in ["library", "mount", "parent", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--suite", choices=["smoke", "matrix", "long"], default="matrix")
    args = p.parse_args()
    if not args.mount.is_mount() or not args.parent.is_dir():
        raise ValueError("native mount/parent required")
    if args.suite == "smoke":
        cases = [(448 << 10, 8, 8, "dense", 1, 1)]
    elif args.suite == "long":
        cases = [(64 << 10, 8, 8, "dense", 4, 128), (64 << 10, 8, 16, "dense", 4, 128)]
    else:
        cases = [
            (size, slots, delay, pat, c, 1)
            for size in [64 << 10, 448 << 10, 1 << 20, 4 << 20]
            for slots, delay, pat, c in [
                (32, 8, "dense", 1),
                (8, 8, "dense", 1),
                (8, 8, "dense", 4),
                (8, 8, "dense", 8),
                (8, 16, "dense", 1),
                (8, 8, "active-hot", 1),
                (1, 8, "dense", 1),
                (0.5, 8, "dense", 1),
                (8, 8, "sparse", 1),
            ]
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for repeat in range(args.repeat):
            for case in cases:
                modes = (
                    ["lru", "drain"]
                    if args.suite == "long"
                    else ["eager", "lru", "drain"]
                )
                modes = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
                for mode in modes:
                    size, slots, delay, pat, c, epochs = case
                    row = run(
                        mode,
                        size,
                        slots,
                        delay,
                        pat,
                        c,
                        args.library,
                        args.mount,
                        args.parent,
                        epochs,
                    )
                    row["repeat"] = repeat
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                print(repeat, case, flush=True)


if __name__ == "__main__":
    main()
