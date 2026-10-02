"""Compare logical KV blocks and immutable policy extents on real 3FS."""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from mooncake.store import MooncakeDistributedStore, ReplicateConfig

from agent_rl_dfs_bench import Buffer
from policy_extent_store import PolicyExtentStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", required=True)
    p.add_argument("--dedicated-master-confirmed", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extent-blocks", type=int, default=1)
    p.add_argument("--block-bytes", type=int, default=458752)
    p.add_argument("--keys", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--read-stride", type=int, default=1)
    p.add_argument("--restore-repeats", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if (
        not a.dedicated_master_confirmed
        or os.environ.get("MOONCAKE_DFS_FS_ADAPTER") != "hf3fs"
    ):
        p.error("requires a dedicated real hf3fs master")
    if (
        min(
            a.keys,
            a.epochs,
            a.extent_blocks,
            a.block_bytes,
            a.read_stride,
            a.restore_repeats,
        )
        < 1
    ):
        p.error("sizes must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "offload").mkdir()
    client = MooncakeDistributedStore()
    rc = client.setup(
        "127.0.0.1:53301",
        "P2PHANDSHAKE",
        0,
        134217728,
        "tcp",
        "",
        a.master,
        enable_ssd_offload=True,
        ssd_offload_path=str(a.output / "offload"),
    )
    assert rc == 0, rc
    replica = ReplicateConfig()
    replica.replica_num = 1
    replica.dfs_replica_num = 1
    buffers = [Buffer(a.block_bytes, a.device) for _ in range(a.keys)]
    for buf in buffers:
        assert client.register_buffer(buf.address, buf.size) == 0
    store = PolicyExtentStore(
        client, replica, a.output.parent.name, a.extent_blocks, a.device
    )
    events, epochs = [], []
    keys = [f"prefix-{i}" for i in range(a.keys)]
    selected = list(range(0, a.keys, a.read_stride))

    def timed(operation, fn, **fields):
        start, cpu = time.perf_counter_ns(), time.process_time_ns()
        value = fn()
        events.append(
            dict(
                operation=operation,
                wall_ns=time.perf_counter_ns() - start,
                cpu_ns=time.process_time_ns() - cpu,
                **fields,
            )
        )
        return value

    cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
    try:
        for epoch in range(a.epochs):
            for i, buf in enumerate(buffers):
                seed = hashlib.sha256(f"{epoch}:{i}".encode()).digest()
                buf.fill((seed * ((buf.size + 31) // 32))[: buf.size])
            expected = [hashlib.sha256(buf.raw()).hexdigest() for buf in buffers]
            mounted = client.allocate_and_mount_segment(268435456)
            assert mounted["ret"] == 0, mounted
            timed(
                "put",
                lambda: store.put(epoch, keys, buffers),
                logical_bytes=a.keys * a.block_bytes,
            )
            live = list(store.extents)
            assert client.batch_is_exist(live) == [1] * len(live)
            # Verify every logical block in memory before the forced DFS miss.
            timed("memory_get", lambda: store.get(epoch, keys, buffers))
            assert [
                hashlib.sha256(buf.raw()).hexdigest() for buf in buffers
            ] == expected
            assert client.unmount_and_free_segment(mounted["segment_ids"], 0) == 0
            assert not any(
                d.is_memory_replica()
                for key in live
                for d in client.get_replica_desc(key)
            )
            before = store.read_bytes
            for _ in range(a.restore_repeats):
                timed(
                    "dfs_get",
                    lambda: store.get(
                        epoch,
                        [keys[i] for i in selected],
                        [buffers[i] for i in selected],
                    ),
                    logical_bytes=len(selected) * a.block_bytes,
                )
                for i in selected:
                    assert hashlib.sha256(buffers[i].raw()).hexdigest() == expected[i]
            transferred = store.read_bytes - before
            retired = timed(
                "reset", lambda: store.advance_policy(epoch + 1), objects=len(live)
            )
            assert client.batch_is_exist(retired) == [0] * len(retired)
            try:
                store.get(epoch, keys[:1], buffers[:1])
            except ValueError:
                pass
            else:
                raise AssertionError("old-policy read accepted")
            epochs.append(
                dict(
                    epoch=epoch,
                    physical_objects=len(live),
                    stale_hits=0,
                    logical_read_bytes=len(selected)
                    * a.block_bytes
                    * a.restore_repeats,
                    dfs_read_bytes=transferred,
                    checksums=expected,
                )
            )
    finally:
        store.close()
        client.close()
    result = dict(
        level="real Mooncake+3FS synthetic policy-extent PoC, not trainer integration",
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        module_path=__import__("mooncake.store", fromlist=["store"]).__file__,
        implementation=str(Path(__file__).resolve()),
        logical_put_bytes=a.epochs * a.keys * a.block_bytes,
        logical_dfs_read_bytes=a.epochs
        * len(selected)
        * a.block_bytes
        * a.restore_repeats,
        wall_ns=time.perf_counter_ns() - wall_start,
        process_cpu_ns=time.process_time_ns() - cpu_start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        lifecycle=epochs,
    )
    for operation in ["put", "dfs_get", "memory_get", "reset"]:
        rows = [row for row in events if row["operation"] == operation]
        result[operation] = dict(
            calls=len(rows),
            wall_ns=sum(row["wall_ns"] for row in rows),
            cpu_ns=sum(row["cpu_ns"] for row in rows),
        )
    (a.output / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "lifecycle"}))


if __name__ == "__main__":
    main()
