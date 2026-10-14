---
orphan: true
---

# Parallel MEMORY and DFS batch puts

The opt-in client path overlaps MEMORY/NoF transfers with DFS staging and synchronous writes on the calling thread. Both paths complete before the existing per-key success decision. It adds no DFS worker and does not acknowledge a partially written key.

Parallel writes require the fenced DFS protocol described below. Legacy RPC signatures and message formats are unchanged. Three additional batch RPCs carry a write-attempt token; applications continue to use the same Store APIs.

## Enablement and compatibility

Start the Master with `MOONCAKE_ENABLE_DFS=1`, `MC_STORE_DFS_WRITE_FENCING=1`, and an **empty, dedicated DFS root**. Initialize each writing client's distributed backend with the matching layout. Enable `MC_STORE_PARALLEL_DFS_PUT=1` before creating clients; it enables fencing on the client automatically. To measure a serial control with the same protocol, set `MC_STORE_DFS_WRITE_FENCING=1` and leave parallel writes disabled.

Both options default to disabled. MEMORY-only operations and ordinary deployments retain the legacy APIs. A fenced namespace rejects tokenless DFS writes, and an opted-in client fails before data transfer when the Master does not support or enable fencing. Readers may use the existing APIs. Descriptor-only DFS put sessions are unsupported in this mode because their API does not retain the attempt context.

`BatchPut` supports plain and `prefer_alloc_in_same_node` merged batches. Fenced single-key Put/Upsert delegate internally to their batch APIs. BatchUpsert retains serial scheduling but drains its transfers before acknowledging completion. TENT, HA, snapshots, and legacy DISK mixtures are unsupported. Sources must be ready and immutable until return; the option does not synchronize an unrelated producer process or add a CUDA stream API.

## Result ordering and I/O ownership

The client submits non-DFS transfers, collects DFS results without changing the transfer summary, drains the transfers, and then applies DFS results. This preserves MEMORY/NoF error precedence and strict replica success for DFS requests. Known submission failures skip DFS. A later MEMORY failure still waits for any started DFS I/O before revocation.

Classic Transfer Engine drain checks every task and completion callback before releasing batch ownership. `BatchCleanupDeferred` alone is not physical completion. Failed partial submissions retain posted work until it terminates. No new end-to-end deadline or cancellation guarantee is introduced; an unobservable completion can keep the synchronous request blocked.

The client retains a shared backend snapshot across DFS I/O. Backend replacement cannot destroy its active owner. Sources in mounted local Store regions use serial scheduling to avoid an overlapping read of a potentially aliased target.

HF3FS read and write operations retain prepared I/O after submit-notification or wait failures. They validate completion count, ring index, userdata, and byte length before shared Iov reuse. Reads also drain before a later write reuses that thread's Iov. Init/Shutdown wait for active positional I/O. An interrupted or empty wait is retried; an unaccounted completion retains ownership until the prepared request can be reaped.

## Fenced writer lifecycle

Each client issues increasing attempt sequences without changing its node `client_id`. The Master records the token under the same lock as admission and validates it under the mutation lock for End/Revoke. Duplicate pending Start is idempotent. Delayed Start, End, and Revoke from a retired attempt cannot touch a later write.

An active fenced writer cannot be preempted, expired, or removed by ordinary replica cleanup. Matching Revoke is the client's acknowledgement that its submitted I/O has drained. Matching End preserves the existing `WriteAt` completion policy; it adds no object-data `fsync` durability guarantee.

A dead or permanently blocked writer therefore retains metadata and allocation rather than authorizing unsafe reuse. There is no automatic orphan GC. Per-shard replay watermarks retain key/client history after deletion and are capped at 4,096 entries per metadata shard. At the limit, new fenced identities fail with `NO_AVAILABLE_HANDLE`; existing identities can continue with increasing sequences. MEMORY-only admission does not use this budget.

The Master persists `.mooncake_fenced_writer_epoch` before exposing allocations. A sealed root cannot be reopened by a later Master, including one with fencing disabled, because allocator recovery and old-client I/O fencing across restart are not implemented. Use a fresh namespace after restart. Do not remove the seal, reuse old files, or downgrade a Master onto that root while old clients, readers, FUSE I/O, or storage requests may remain active. This seal is a conservative reuse barrier, not allocator persistence or HA recovery.

## Tests and measurements

CPU ordering tests use the production Client, in-process Master, memcpy worker, DFS staging, and POSIX backend with a host-backed device double. They cover serial/parallel and merged paths, all success/failure combinations, commit/revoke barriers, and backend ownership. Protocol tests exercise late Start/End/Revoke, idempotency, expiry, preemption, restart rejection, and the replay-history budget.

The HF3FS C API double links the production adapter/resource manager and tests prepare/submit/wait errors, malformed and short completions, read-to-write Iov reuse, and concurrent Shutdown. Actual HF3FS tests verify bytes on a real mount. The CUDA/native client test verifies an immutable GPU source in both MEMORY and DFS.

```bash
cmake --build build --target dfs_sync_client_test transfer_task_test \
  transfer_submitter_config_test hf3fs_write_lifetime_test dfs_hf3fs_test
MOONCAKE_NATIVE_TEST_ROOT="$HF3FS_TEST_ROOT" \
  ./build/mooncake-store/tests/dfs_sync_client_test
./build/mooncake-store/tests/transfer_task_test
./build/mooncake-store/tests/hf3fs_write_lifetime_test
```

The native root must be writable and isolated. A filtered test that runs zero cases, or a skipped CUDA/native test, does not establish correctness.

`parallel_dfs_put_bench` compares the same fenced protocol with serial or parallel scheduling. It uses an in-process Master, local MEMORY, and native USRBIO, verifies a DFS readback, and reports PUT P50/P95/P99, logical throughput, process CPU time, and peak RSS. Its loop throughput includes per-batch removal; PUT latency excludes removal. Setup and namespace cleanup are outside both measurement windows. This is not remote RDMA/NoF contention, cold-storage, or trainer measurement.

```bash
cmake --build build --target parallel_dfs_put_bench
./build/mooncake-store/benchmarks/parallel_dfs_put_bench \
  "$HF3FS_TEST_ROOT/run-serial" serial 2000 1 1048576 gpu
./build/mooncake-store/benchmarks/parallel_dfs_put_bench \
  "$HF3FS_TEST_ROOT/run-parallel" parallel 2000 1 1048576 gpu
```

The benchmark refuses existing namespaces and removes only its own successful run after all owners stop. Use paired repetitions with alternating order. Report contention, retained storage, and negative results rather than assuming overlap improves latency.

## Measured local-MEMORY result

One GPU client, local MEMORY, a native 3FS DFS replica, 1 MiB values, batch one, and 2,000 PUT samples per run were compared in five paired repetitions with alternating order. Both variants used the same Release/CUDA/3FS binary and fenced protocol. PUT latency excludes payload verification. Process CPU and loop wall time include one DFS readback and per-batch removal; tracing was performed in separate runs. The table gives the median of each variant's run statistics; paired ratios are computed per repetition.

| Measure | Serial median | Parallel median | Median paired parallel/serial ratio |
|---|---:|---:|---:|
| PUT P50 | 11.277 ms | 11.167 ms | 0.990 |
| PUT P95 | 11.882 ms | 11.828 ms | 0.996 |
| PUT P99 | 14.259 ms | 13.457 ms | 0.932 |
| Logical PUT throughput | 0.08550 GiB/s | 0.08615 GiB/s | 1.010 |
| Process CPU time | 0.839 s | 0.873 s | 1.047 |
| Peak RSS | 387,384 KiB | 387,384 KiB | 1.000 |

The effect is small and unstable: one pair's P95/P99 ratios were 1.563/1.475, despite lower medians across runs. These results do not meet a 10% P95 improvement criterion or establish a stable tail-latency win. Keep the feature disabled for this workload. Remote transfer, NIC contention, NoF, tensor parallelism, and trainer speedup require separate measurements.

A separate 50-PUT API trace observed no serial overlap and overlap in all 50 parallel requests. Median MEMORY D2H API spans were approximately 74/98 microseconds for serial/parallel, DFS D2H spans 26/27 microseconds, and native write-completion waits 10.9/11.1 milliseconds. This explains the small upper bound on hiding local MEMORY work. API wall spans include CUDA blocking and do not prove simultaneous GPU DMA or quantify PCIe/NIC utilization.
