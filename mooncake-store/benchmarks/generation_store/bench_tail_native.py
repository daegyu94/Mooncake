"""Native 3FS write combining and policy-expiry tail admission A/B/C.

immediate: prior generation prototype, synchronous append.
combine: tail buffering, explicit full DFS drain after live reads.
expire: same buffering, revoke after live reads and discard unneeded tail.
The last two share the same RAM budget and differ only at terminal policy fence.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import uuid
from contextlib import nullcontext

from bench_generation import timed
from generation_store import GenerationStore, NativeIO
from tail_store import TailBufferedStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--suite", choices=["smoke", "matrix", "physical", "long"], required=True
    )
    p.add_argument("--probe-module", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    io = NativeIO(a.library, a.mount, depth=8)
    rows = []
    if a.suite == "physical":
        sys.path.insert(0, str(a.probe_module))
        probe = importlib.import_module("storage_aio_probe").StorageIO
    else:
        probe = None
    cases = [(448 << 10, 3, 1, 4 << 20, 1)]
    if a.suite == "matrix":
        cases = [
            (sz, n, 1, 4 << 20, 1)
            for sz in [64 << 10, 448 << 10, 1 << 20, 4 << 20]
            for n in [3, 17]
        ]
    elif a.suite == "physical":
        cases = [(448 << 10, n, 1, 4 << 20, 1) for n in [3, 127]]
        cases += [(448 << 10, 3, 8, 256 << 10, 8)]
    elif a.suite == "long":
        cases = [
            (448 << 10, 3, 128, budget, overlap)
            for budget, overlap in [(4 << 20, 1), (4 << 20, 4), (256 << 10, 8)]
        ]
    repetitions = 1 if a.suite == "smoke" else 5
    for size, count, epochs, budget, overlap in cases:
        for repeat in range(repetitions):
            variants = ["immediate", "combine", "expire"]
            variants = variants[repeat % 3 :] + variants[: repeat % 3]
            for variant in variants:
                name = f"s{size}-n{count}-e{epochs}-b{budget}-o{overlap}-r{repeat}-{variant}"
                directory = a.output / name
                directory.mkdir()
                root = a.mount / "verl-lab" / ("tail-" + uuid.uuid4().hex)
                store = (
                    GenerationStore(io, root, 1 << 30)
                    if variant == "immediate"
                    else TailBufferedStore(io, root, 1 << 30, tail_budget=budget)
                )
                before = dict(io.stats)
                events = []
                live = []
                phase = dict(
                    append_ns=0, read_ns=0, drain_ns=0, transition_ns=0, gc_ns=0
                )
                context = probe(directory) if probe else nullcontext()
                with context:
                    cpu = time.process_time_ns()
                    wall = time.perf_counter_ns()
                    for epoch in range(epochs):
                        g = store.create()
                        value = bytes([epoch % 251]) * size
                        for _ in range(count):
                            _, t = timed(lambda: store.append(g, [value]))
                            phase["append_ns"] += t["wall_ns"]
                        store.seal(g)
                        data, t = timed(lambda: store.get(g, list(range(count))))
                        phase["read_ns"] += t["wall_ns"]
                        assert data == [value] * count
                        if variant == "combine":
                            _, t = timed(lambda: store.flush(g))
                            phase["drain_ns"] += t["wall_ns"]
                        live.append((g, epoch))
                        if len(live) > overlap:
                            old, _ = live.pop(0)
                            _, t = timed(lambda: store.invalidate(old))
                            phase["transition_ns"] += t["wall_ns"]
                            with_error = False
                            try:
                                store.get(old, [0])
                            except KeyError:
                                with_error = True
                            assert with_error
                        # Revisit every retained policy; identical requested bytes.
                        for retained, version in live:
                            data, t = timed(lambda: store.get(retained, [count - 1]))
                            phase["read_ns"] += t["wall_ns"]
                            assert data == [bytes([version % 251]) * size]
                        if epoch % 8 == 7:
                            _, t = timed(store.gc)
                            phase["gc_ns"] += t["wall_ns"]
                        events.append(
                            dict(
                                epoch=epoch,
                                live=len(store.generations),
                                retired=len(store.retired),
                                resident_tail_bytes=getattr(
                                    store, "resident_tail_bytes", 0
                                ),
                                fd_count=len(list(Path("/proc/self/fd").iterdir())),
                            )
                        )
                    for g, _ in live:
                        _, t = timed(lambda: store.invalidate(g))
                        phase["transition_ns"] += t["wall_ns"]
                    _, t = timed(store.gc)
                    phase["gc_ns"] += t["wall_ns"]
                    elapsed = time.perf_counter_ns() - wall
                    consumed = time.process_time_ns() - cpu
                delta = {k: io.stats[k] - before[k] for k in io.stats}
                logical = size * count * epochs
                discarded = store.metrics.get("discarded_tail_bytes", 0)
                assert delta["write_bytes"] + discarded == logical
                assert not store.generations and not store.retired
                assert getattr(store, "resident_tail_bytes", 0) == 0
                rows.append(
                    dict(
                        name=name,
                        variant=variant,
                        size=size,
                        count=count,
                        epochs=epochs,
                        repeat=repeat,
                        budget=budget,
                        overlap=overlap,
                        wall_ns=elapsed,
                        cpu_ns=consumed,
                        logical_put_bytes=logical,
                        logical_put_count=count * epochs,
                        io=delta,
                        metrics=store.metrics,
                        phases=phase,
                        events=events,
                    )
                )
                (a.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    name,
                    f"wall={elapsed / 1e9:.4f}s write={delta['write_bytes']} discard={discarded}",
                    flush=True,
                )
    manifest = dict(
        suite=a.suite,
        bridge_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        validation="Native 3FS independent research path; no production Mooncake/GPU/trainer",
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        loaded_libraries=[
            s
            for s in Path("/proc/self/maps").read_text().splitlines()
            if "usrbio_bridge" in s or "libhf3fs_api" in s
        ],
        copy_scope="bridge memcpy plus separate explicit tail copies; not total physical copies",
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    io.close()


if __name__ == "__main__":
    main()
