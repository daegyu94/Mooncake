#!/usr/bin/env python3
"""Deterministic KV lifecycle benchmark against an explicitly dedicated master.

CPU and optional CUDA buffers exercise the real Store/DFS adapter. This is
a synthetic lifecycle benchmark, independent of trainer scheduling. Keep the master and filesystem namespace exclusive to this
run: invalidation intentionally removes all keys in that master.
"""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mooncake.store import MooncakeDistributedStore, ReplicateConfig


class Buffer:
    """Address-stable CPU or optional CUDA fixture, with checks outside I/O timing."""

    def __init__(self, size, device):
        self.size = size
        self.torch = None
        if device == "cpu":
            self.data = ctypes.create_string_buffer(size)
            self.address = ctypes.addressof(self.data)
        else:
            import torch

            self.torch = torch
            self.data = torch.empty(size, dtype=torch.uint8, device=device)
            self.address = self.data.data_ptr()

    def fill(self, payload):
        if self.torch is None:
            ctypes.memmove(self.address, payload, self.size)
        else:
            self.data.copy_(
                self.torch.frombuffer(bytearray(payload), dtype=self.torch.uint8)
            )

    def clear(self):
        if self.torch is None:
            ctypes.memset(self.address, 0, self.size)
        else:
            self.data.zero_()

    def raw(self):
        return (
            self.data.raw if self.torch is None else self.data.cpu().numpy().tobytes()
        )

    def synchronize(self):
        if self.torch is not None:
            self.torch.cuda.synchronize(self.data.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument(
        "--device", default="cpu", help="cpu or an optional CUDA device such as cuda:0"
    )
    parser.add_argument("--dedicated-master-confirmed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-bytes", type=int, default=458752)
    parser.add_argument("--keys", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--policy-key-mode", choices=["epoch", "reuse"], default="epoch"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-precheck", choices=["on", "off"], default="off")
    parser.add_argument(
        "--warm-fraction", type=float, choices=[0.0, 0.5, 1.0], default=0.0
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--slices", type=int, default=56)
    parser.add_argument("--local-buffer-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--segment-bytes", type=int, default=268435456)
    args = parser.parse_args()
    if not args.dedicated_master_confirmed:
        parser.error("Use an exclusive master; invalidation removes all its keys")
    if (
        min(
            args.payload_bytes,
            args.keys,
            args.epochs,
            args.batch_size,
            args.concurrency,
            args.slices,
        )
        <= 0
        or args.slices > args.payload_bytes
    ):
        parser.error("Sizes must be positive and slices cannot exceed payload bytes")
    if args.payload_bytes * args.keys > args.segment_bytes // 2:
        parser.error("Live logical data must fit comfortably inside the memory tier")
    if os.environ.get("MOONCAKE_DFS_FS_ADAPTER") != "hf3fs":
        parser.error("This run requires the real hf3fs adapter")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh run")
    args.output.mkdir(parents=True)
    (args.output / "local-offload").mkdir()
    events = []
    buffers = [Buffer(args.payload_bytes, args.device) for _ in range(args.keys)]
    base, remainder = divmod(args.payload_bytes, args.slices)
    sizes = [base + (index < remainder) for index in range(args.slices)]
    starts = [sum(sizes[:index]) for index in range(args.slices)]
    pointers = [[buffer.address + offset for offset in starts] for buffer in buffers]
    all_sizes = [sizes for _ in buffers]
    replica = ReplicateConfig()
    replica.replica_num = 1
    replica.dfs_replica_num = 1
    port = 52300

    def setup():
        nonlocal port
        port += 1
        client = MooncakeDistributedStore()
        rc = client.setup(
            f"127.0.0.1:{port}",
            "P2PHANDSHAKE",
            0,
            args.local_buffer_bytes,
            "tcp",
            "",
            args.master,
            enable_ssd_offload=True,
            ssd_offload_path=str(args.output / "local-offload"),
        )
        if rc != 0:
            raise RuntimeError(f"setup failed: {rc}")
        for buffer in buffers:
            rc = client.register_buffer(buffer.address, args.payload_bytes)
            if rc != 0:
                raise RuntimeError(f"register_buffer failed: {rc}")
        return client

    def timed(name, fn, **fields):
        wall = time.perf_counter_ns()
        cpu = time.process_time_ns()
        result = fn()
        events.append(
            dict(
                operation=name,
                wall_ns=time.perf_counter_ns() - wall,
                cpu_ns=time.process_time_ns() - cpu,
                t_ns=time.monotonic_ns(),
                **fields,
            )
        )
        return result

    def batches(client, operation, keys):
        buffers[0].synchronize()

        def batch(start):
            stop = min(start + args.batch_size, len(keys))

            def fn():
                if operation in ("put", "warm_put"):
                    indices = list(range(start, stop))
                    if operation == "put" and args.save_precheck == "on":
                        states = timed(
                            "save_exists",
                            lambda: client.batch_is_exist(keys[start:stop]),
                            keys=stop - start,
                        )
                        if len(states) != stop - start or any(v < 0 for v in states):
                            raise RuntimeError(f"Existence query failed: {states}")
                        indices = [i for i in indices if states[i - start] != 1]
                    if indices:
                        transferred = timed(
                            "native_put",
                            lambda: client.batch_put_from_multi_buffers(
                                [keys[i] for i in indices],
                                [pointers[i] for i in indices],
                                [all_sizes[i] for i in indices],
                                replica,
                            ),
                            keys=len(indices),
                            phase=operation,
                        )
                        if len(transferred) != len(indices) or any(transferred):
                            raise RuntimeError(f"Native put failed: {transferred}")
                    result = [0] * (stop - start)
                else:
                    result = client.batch_get_into_multi_buffers(
                        keys[start:stop], pointers[start:stop], all_sizes[start:stop]
                    )
                buffers[0].synchronize()
                return result

            results = timed(
                operation,
                fn,
                keys=stop - start,
                logical_bytes=(stop - start) * args.payload_bytes,
            )
            # put returns zero on success; get returns the completed byte count.
            expected = 0 if operation in ("put", "warm_put") else args.payload_bytes
            if len(results) != stop - start or any(rc != expected for rc in results):
                raise RuntimeError(
                    f"{operation} failed: {results}, expected {expected}"
                )

        def submit():
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(batch, range(0, len(keys), args.batch_size)))

        timed(
            operation + "_phase",
            submit,
            keys=len(keys),
            logical_bytes=len(keys) * args.payload_bytes,
        )

    def checksum():
        return [hashlib.sha256(buffer.raw()).hexdigest() for buffer in buffers]

    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    lifecycle = []
    producer = setup()
    try:
        for epoch in range(args.epochs):
            policy_key = epoch if args.policy_key_mode == "epoch" else "reused"
            keys = [
                f"agent-rl-policy-{policy_key}-prefix-{i}" for i in range(args.keys)
            ]
            for i, buffer in enumerate(buffers):
                seed = hashlib.sha256(f"policy={epoch},prefix={i}".encode()).digest()
                payload = (seed * ((args.payload_bytes + 31) // 32))[
                    : args.payload_bytes
                ]
                buffer.fill(payload)
            expected = checksum()
            mounted = producer.allocate_and_mount_segment(args.segment_bytes)
            if mounted["ret"] != 0:
                raise RuntimeError(f"Segment allocation failed: {mounted}")
            warm_keys = int(args.keys * args.warm_fraction)
            if warm_keys:
                batches(producer, "warm_put", keys[:warm_keys])
                # Conflicting duplicate payloads must never replace published KV.
                for buffer in buffers[:warm_keys]:
                    buffer.fill(bytes([0xA5]) * args.payload_bytes)
            batches(producer, "put", keys)
            hits = timed(
                "same_policy_exists",
                lambda client=producer, batch_keys=keys: client.batch_is_exist(
                    batch_keys
                ),
                keys=len(keys),
            )
            if hits != [1] * len(keys):
                raise RuntimeError(f"Existing KV keys missing: {hits}")
            for buffer in buffers:
                buffer.clear()
            batches(producer, "memory_get", keys)
            if checksum() != expected:
                raise RuntimeError("Memory-tier KV checksum mismatch")
            before = [
                sum(d.is_memory_replica() for d in producer.get_replica_desc(k))
                for k in keys
            ]
            rc = timed(
                "unmount",
                lambda client=producer,
                ids=mounted["segment_ids"]: client.unmount_and_free_segment(ids, 0),
            )
            if rc != 0:
                raise RuntimeError(f"Unmount failed: {rc}")
            after = [
                sum(d.is_memory_replica() for d in producer.get_replica_desc(k))
                for k in keys
            ]
            if any(after):
                raise RuntimeError(
                    f"Memory replicas survived explicit unmount: {after}"
                )
            for buffer in buffers:
                buffer.clear()
            batches(producer, "dfs_get", keys)
            if checksum() != expected:
                raise RuntimeError("DFS-tier KV checksum mismatch")
            removed = timed(
                "reset", lambda client=producer: client.remove_all(force=True)
            )
            stale = timed(
                "stale_check",
                lambda client=producer, batch_keys=keys: client.batch_is_exist(
                    batch_keys
                ),
                keys=len(keys),
            )
            if removed < 0 or any(v != 0 for v in stale):
                raise RuntimeError(
                    f"Invalidation failed: removed={removed}, states={stale}"
                )
            lifecycle.append(
                {
                    "epoch": epoch,
                    "memory_replicas_before": before,
                    "memory_replicas_after": after,
                    "removed": removed,
                    "stale_hits": sum(v == 1 for v in stale),
                    "checksums": expected,
                }
            )
    finally:
        producer.close()
        del producer
    summary = {
        "level": f"real Mooncake+3FS; synthetic {args.device} KV lifecycle",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "module_path": __import__("mooncake.store", fromlist=["store"]).__file__,
        "configuration": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("MOONCAKE_DFS_")
        },
        "wall_ns": time.perf_counter_ns() - wall_start,
        "process_cpu_ns": time.process_time_ns() - cpu_start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "staging_pool_bytes": int(
            os.environ.get("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "1342177280")
        ),
        "logical_put_bytes": args.payload_bytes * args.keys * args.epochs,
        "logical_dfs_read_bytes": args.payload_bytes * args.keys * args.epochs,
        "lifecycle": lifecycle,
    }
    for operation in sorted({event["operation"] for event in events}):
        selected = [event for event in events if event["operation"] == operation]
        summary[operation] = {
            "calls": len(selected),
            "wall_ns": sum(e["wall_ns"] for e in selected),
            "median_wall_ns": statistics.median(e["wall_ns"] for e in selected),
            "cpu_ns": sum(e["cpu_ns"] for e in selected),
            "keys": sum(e.get("keys", 0) for e in selected),
        }
    (args.output / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "lifecycle"}), flush=True
    )


if __name__ == "__main__":
    main()
