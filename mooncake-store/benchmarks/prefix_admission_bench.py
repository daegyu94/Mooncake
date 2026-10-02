#!/usr/bin/env python3
"""Native Store metadata PoC for a stable single-group full-attention prefix.

This is not a vLLM connector patch. Every block contains all required rank keys.
There is no cross-call cache, and callers must supply policy-qualified keys.
"""

import argparse
import ctypes
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def prefix_length(query, blocks, probe_blocks=0):
    """Return consecutive fully present blocks, never infer suffix presence."""
    if probe_blocks < 0:
        raise ValueError("probe_blocks must be nonnegative")
    if not blocks:
        return 0
    boundary = min(probe_blocks, len(blocks)) if probe_blocks else len(blocks)
    for start, end in [(0, boundary), (boundary, len(blocks))]:
        if start == end:
            continue
        keys = [key for block in blocks[start:end] for key in block]
        result = query(keys)
        if len(result) != len(keys):
            raise ValueError("existence response size mismatch")
        pos = 0
        for i in range(start, end):
            width = len(blocks[i])
            if not width:
                raise ValueError("each block must have required rank keys")
            if any(value != 1 for value in result[pos : pos + width]):
                return i
            pos += width
    return len(blocks)


def metrics(url):
    text = urllib.request.urlopen(url, timeout=5).read().decode()
    names = [
        "master_batch_exist_key_requests_total",
        "master_batch_exist_key_items_total",
    ]
    return {
        name: sum(
            float(line.split()[-1])
            for line in text.splitlines()
            if line.startswith((name + " ", name + "{"))
        )
        for name in names
    }


def main():
    import mooncake.store
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    args.output.mkdir(exist_ok=False)
    store = MooncakeDistributedStore()
    assert (
        store.setup(
            "127.0.0.1:53301", "P2PHANDSHAKE", 0, 16 << 20, "tcp", "", args.master
        )
        == 0
    )
    payload_bytes = 448 * 1024
    buffer = ctypes.create_string_buffer(payload_bytes)
    address = ctypes.addressof(buffer)
    assert store.register_buffer(address, payload_bytes) == 0
    segment = store.allocate_and_mount_segment(256 << 20)
    assert segment["ret"] == 0
    config = ReplicateConfig()
    config.replica_num = 1
    keys = [f"{args.namespace}-policy0-{i:04d}-" + "a" * 64 for i in range(256)]
    for start in range(0, len(keys), 16):
        batch = keys[start : start + 16]
        assert store.batch_put_from_multi_buffers(
            batch, [[address]] * len(batch), [[payload_bytes]] * len(batch), config
        ) == [0] * len(batch)
    key_width = len(keys[0].encode())
    assert all(len(key.encode()) == key_width for key in keys)
    rows = []
    try:
        for case, hit in [("cold", 0), ("short", 8), ("warm", 256)]:
            blocks = [
                [key if i < hit else key.replace("policy0", "policy1")]
                for i, key in enumerate(keys)
            ]
            for repeat in range(3):
                for probe in [0, 16] if repeat % 2 == 0 else [16, 0]:
                    assert prefix_length(store.batch_is_exist, blocks, probe) == hit
                    before = metrics(args.metrics)
                    counters = {"calls": 0, "keys": 0, "key_utf8_bytes": 0}

                    def query(batch, counters=counters):
                        counters["calls"] += 1
                        counters["keys"] += len(batch)
                        counters["key_utf8_bytes"] += len(batch) * key_width
                        return store.batch_is_exist(batch)

                    durations = []
                    cpu = time.process_time_ns()
                    for _ in range(args.iterations):
                        t = time.perf_counter_ns()
                        assert prefix_length(query, blocks, probe) == hit
                        durations.append(time.perf_counter_ns() - t)
                    cpu = time.process_time_ns() - cpu
                    after = metrics(args.metrics)
                    delta = {k: after[k] - before[k] for k in before}
                    assert (
                        delta["master_batch_exist_key_requests_total"]
                        == counters["calls"]
                    )
                    assert (
                        delta["master_batch_exist_key_items_total"] == counters["keys"]
                    )
                    rows.append(
                        dict(
                            case=case,
                            hit_blocks=hit,
                            repeat=repeat,
                            probe=probe,
                            iterations=args.iterations,
                            **counters,
                            master_delta=delta,
                            cpu_ns=cpu,
                            median_ns=statistics.median(durations),
                            p95_ns=sorted(durations)[int(0.95 * (len(durations) - 1))],
                            durations_ns=durations,
                        )
                    )
                    (args.output / "rows.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
                    print(case, repeat, probe, counters, flush=True)
        module = Path(mooncake.store.__file__)
        (args.output / "manifest.json").write_text(
            json.dumps(
                dict(
                    level="Native Mooncake TCP loopback metadata benchmark; no DFS or trainer",
                    module_path=str(module),
                    module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                    payload_bytes=payload_bytes,
                    live_keys=len(keys),
                    rf=1,
                    args={k: str(v) for k, v in vars(args).items()},
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        store.unmount_and_free_segment(segment["segment_ids"], 0)
        store.unregister_buffer(address)
        store.close()


if __name__ == "__main__":
    main()
