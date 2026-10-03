"""Native true-range versus bounded generation-local coalescing.

Equal layout, buffers, depth and logical reads across A/B/C. Eight groups form
one fixed workload; concurrency changes scheduling, not total operations.
Client read-count counters are not network RPC counters.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import uuid

from generation_store import NativeIO
from range_planner import PlannedStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", choices=["smoke", "matrix", "physical"], required=True)
    p.add_argument("--probe-module", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    writer = NativeIO(a.library, a.mount, depth=8)
    channels = [NativeIO(a.library, a.mount, depth=8) for _ in range(8)]
    # Keep registered capacity equal even for concurrency=1 and exact ranges.
    probe = None
    if a.suite == "physical":
        sys.path.insert(0, str(a.probe_module))
        probe = importlib.import_module("storage_aio_probe").StorageIO
    sizes = [64 << 10, 448 << 10, 1 << 20, 4 << 20]
    if a.suite != "matrix":
        sizes = [448 << 10]
    rows = []
    for size in sizes:
        root = a.mount / "verl-lab" / ("planner-" + uuid.uuid4().hex)
        store = PlannedStore(writer, root, 1 << 30)
        g = store.create()
        values = [bytes([k]) * size for k in range(64)]
        store.append(g, values)
        store.seal(g)
        for concurrency in [1, 8] if a.suite == "matrix" else [1]:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for pattern in ["dense", "sparse", "holes"]:
                    groups = [
                        [
                            base + k
                            for k in range(8)
                            if pattern == "dense"
                            or (pattern == "sparse" and k == 0)
                            or (pattern == "holes" and k != 4)
                        ]
                        for base in range(0, 64, 8)
                    ]
                    for repeat in range(1 if a.suite == "smoke" else 5):
                        variants = ["exact", "adjacent", "gap125"]
                        variants = variants[repeat % 3 :] + variants[: repeat % 3]
                        for variant in variants:
                            name = (
                                f"s{size}-c{concurrency}-{pattern}-r{repeat}-{variant}"
                            )
                            directory = a.output / name
                            directory.mkdir()
                            before = {
                                k: sum(c.stats[k] for c in channels)
                                for k in channels[0].stats
                            }
                            metrics_before = dict(store.metrics)

                            def read_group(index):
                                start = time.perf_counter_ns()
                                keys = groups[index]
                                if variant == "exact":
                                    result = store.get(g, keys, channel=channels[index])
                                else:
                                    result = store.get_planned(
                                        g,
                                        keys,
                                        channel=channels[index],
                                        amplification=1.25
                                        if variant == "gap125"
                                        else 1.0,
                                    )
                                assert result == [values[k] for k in keys]
                                return time.perf_counter_ns() - start

                            context = probe(directory) if probe else nullcontext()
                            with context:
                                usage_before = resource.getrusage(resource.RUSAGE_SELF)
                                cpu = time.process_time_ns()
                                wall = time.perf_counter_ns()
                                request_ns = list(pool.map(read_group, range(8)))
                                elapsed = time.perf_counter_ns() - wall
                                consumed = time.process_time_ns() - cpu
                                usage_after = resource.getrusage(resource.RUSAGE_SELF)
                            delta = {
                                k: sum(c.stats[k] for c in channels) - before[k]
                                for k in before
                            }
                            requested = size * sum(map(len, groups))
                            assert delta["read_bytes"] >= requested
                            assert delta["read_bytes"] <= requested * (
                                1.25 if variant == "gap125" else 1
                            )
                            rows.append(
                                dict(
                                    name=name,
                                    size=size,
                                    concurrency=concurrency,
                                    pattern=pattern,
                                    repeat=repeat,
                                    variant=variant,
                                    wall_ns=elapsed,
                                    cpu_ns=consumed,
                                    requested_bytes=requested,
                                    request_ns=request_ns,
                                    io=delta,
                                    metrics={
                                        k: v - metrics_before.get(k, 0)
                                        for k, v in store.metrics.items()
                                    },
                                    voluntary_switches=usage_after.ru_nvcsw
                                    - usage_before.ru_nvcsw,
                                    involuntary_switches=usage_after.ru_nivcsw
                                    - usage_before.ru_nivcsw,
                                    fd_count=len(list(Path("/proc/self/fd").iterdir())),
                                )
                            )
                            (a.output / "rows.json").write_text(
                                json.dumps(rows, indent=2) + "\n"
                            )
                            print(
                                name,
                                f"wall={elapsed / 1e9:.4f}s reads={delta['requests']} bytes={delta['read_bytes']}",
                                flush=True,
                            )
        store.invalidate(g)
        assert store.gc() == 1
    (a.output / "manifest.json").write_text(
        json.dumps(
            dict(
                bridge_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
                validation="Native 3FS independent research path; no production Mooncake/GPU/trainer",
                depth=8,
                slot_bytes=4 << 20,
                reader_channels=8,
                reader_registered_bytes=8 * 8 * (4 << 20),
                cache_state="No eviction; rotated variants after one-time write. Mixed/warm service state, not cold-cache experiment.",
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                loaded_libraries=[
                    s
                    for s in Path("/proc/self/maps").read_text().splitlines()
                    if "usrbio_bridge" in s or "libhf3fs_api" in s
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    for channel in channels:
        channel.close()
    writer.close()


if __name__ == "__main__":
    main()
