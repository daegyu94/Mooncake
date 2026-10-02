"""Deterministic native 3FS PoC: range A/B, generations, compact index."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import resource
import time
import uuid
from generation_store import GenerationStore, NativeIO


def timed(fn):
    w = time.perf_counter_ns()
    c = time.process_time_ns()
    value = fn()
    return value, dict(
        wall_ns=time.perf_counter_ns() - w, cpu_ns=time.process_time_ns() - c
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--long-epochs", type=int, default=128)
    p.add_argument("--keys", type=int, default=128)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    namespace = a.mount / "verl-lab" / ("generation-" + uuid.uuid4().hex)
    owner = NativeIO(a.library, a.mount, depth=8)
    readers = [NativeIO(a.library, a.mount, depth=1) for _ in range(8)]
    store = GenerationStore(owner, namespace, 1 << 30)
    rows = []
    lifecycle = []

    def save():
        (a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
        (a.output / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")

    for size in [64 << 10, 448 << 10, 1 << 20, 4 << 20]:
        g = store.create()
        values = [
            (hashlib.sha256(f"{size}:{i}".encode()).digest() * ((size + 31) // 32))[
                :size
            ]
            for i in range(a.keys)
        ]
        store.append(g, values)
        store.seal(g)
        for concurrency in [1, 4, 8]:
            # Partition by extent, so each A/B fetches an extent at most once.
            for pattern, keys in [
                ("dense", list(range(a.keys))),
                ("sparse", list(range(0, a.keys, 16))),
            ]:
                groups = [
                    [k for k in keys if k // 16 == extent]
                    for extent in range((a.keys + 15) // 16)
                ]
                for repeat in range(5):
                    for whole in [16, 0] if repeat % 2 == 0 else [0, 16]:
                        start = [dict(r.stats) for r in readers]

                        def work(i):
                            return store.get(
                                g, groups[i], whole_extent=whole, channel=readers[i]
                            )

                        # One task per extent; no duplicate whole-extent fetch across clients.
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            result, measure = timed(
                                lambda: list(pool.map(work, range(len(groups))))
                            )
                        for i, actual in enumerate(result):
                            assert actual == [values[k] for k in groups[i]]
                        delta = {
                            k: sum(r.stats[k] - s[k] for r, s in zip(readers, start))
                            for k in readers[0].stats
                        }
                        rows.append(
                            dict(
                                size=size,
                                concurrency=concurrency,
                                active_clients=min(len(groups), concurrency),
                                pattern=pattern,
                                repeat=repeat,
                                path="whole" if whole else "range",
                                requested_bytes=len(keys) * size,
                                **measure,
                                **delta,
                            )
                        )
                save()
        store.invalidate(g)
        store.gc()
    # Native multi-generation turnover, same payload, total occupied-byte cap.
    for overlap in [1, 2, 4]:
        retained = []
        for epoch in range(a.long_epochs):
            g = store.create()
            value = bytes([epoch % 256]) * (64 << 10)
            store.append(g, [value] * 8)
            store.seal(g)
            retained.append(g)
            assert store.get(g, [0]) == [value]
            before = time.perf_counter_ns()
            retired = None
            if len(retained) > overlap:
                retired = retained.pop(0)
                store.invalidate(retired)
            transition = time.perf_counter_ns() - before
            _, gc = timed(store.gc)
            if retired is not None:
                try:
                    store.get(retired, [0])
                    raise AssertionError("stale accepted")
                except KeyError:
                    pass
                try:
                    store.append(retired, [b"late"])
                    raise AssertionError("late PUT accepted")
                except KeyError:
                    pass
            lifecycle.append(
                dict(
                    overlap=overlap,
                    epoch=epoch,
                    transition_ns=transition,
                    gc_ns=gc["wall_ns"],
                    gc_cpu_ns=gc["cpu_ns"],
                    live_bytes=sum(x.cursor for x in store.generations.values()),
                    retired_bytes=sum(x.cursor for x in store.retired.values()),
                    fd_count=len(list(Path("/proc/self/fd").iterdir())),
                    live_generations=len(store.generations),
                )
            )
        for g in retained:
            store.invalidate(g)
        store.gc()
        save()
    # Index cost is measured separately by bench_pressure.py (deduplicated deep size).
    manifest = dict(
        namespace=str(namespace),
        validation_level="Native 3FS USRBIO independent research path; not native Mooncake master or trainer",
        bridge_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        copy_scope="bridge memcpy bytes only; Python packing/copy remains",
        owner_stats=owner.stats,
        store_metrics=store.metrics,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        policy_epochs=a.long_epochs * 3,
        loaded_libraries=[
            line.strip()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "usrbio_bridge" in line or "libhf3fs_api" in line
        ],
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for r in readers:
        r.close()
    owner.close()
    print(
        "Completed",
        len(rows),
        "range rows and",
        len(lifecycle),
        "native generation epochs",
        flush=True,
    )


if __name__ == "__main__":
    main()
