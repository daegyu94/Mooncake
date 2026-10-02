# Agent RL KV lifecycle benchmark

`agent_rl_dfs_bench.py` uses the real native Store API and the `hf3fs` adapter with deterministic CPU buffers.
It covers same-policy existence checks and memory-tier reuse, explicit memory segment removal, DFS-only restoration with SHA-256 checks, and drained policy invalidation.
The default CPU fixture does not measure GPU staging, veRL scheduling, or total rollout performance.
Use `--device cuda:0` with an appropriate PyTorch installation to validate GPU staging and byte correctness separately; this still is a synthetic workload.

## Run

Use a separate master and a new directory on an existing 3FS client mount.
The benchmark removes every key in its master; never point it at a production or shared master.
Install a Mooncake wheel built with USRBIO support and matching `libhf3fs_api_shared.so` in an isolated environment.

```bash
export LD_LIBRARY_PATH="$HF3FS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=hf3fs
export MOONCAKE_DFS_ALLOCATOR=shard
export MOONCAKE_DFS_ROOT_DIR="$THREEFS_MOUNT/agent-rl-new-run"
export MOONCAKE_DFS_SHARD_COUNT=4
export MOONCAKE_DFS_SHARD_CAPACITY=268435456
export MOONCAKE_DFS_BUCKET_CAPACITY=268435456
export MOONCAKE_DFS_MAX_BUCKET_COUNT=4
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=true
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=67108864
mkdir -p "$MOONCAKE_DFS_ROOT_DIR"
mooncake_master --port=52151 --metrics_port=19203 --enable_offload=true
```

Run the workload in another terminal with the same environment, then stop only this master.

```bash
python mooncake-store/benchmarks/agent_rl_dfs_bench.py \
  --master 127.0.0.1:52151 --dedicated-master-confirmed \
  --output results/new-run --payload-bytes 458752 \
  --keys 16 --epochs 4 --batch-size 4 --concurrency 1 --slices 56
```

The 448 KiB example matches the observed 16-token block of a Qwen2.5-1.5B workload with 28 layers, two KV heads, dimension 128 and BF16 K/V.
It does not assert a specific Kimi model layout.
Vary payload bytes independently of model size, and use concurrency 1 and 4 to expose serialization costs.
Trainer names such as `separate_async` cannot be assigned to these synthetic concurrency settings.

## Interpretation

Keep total DFS capacity, live logical data, epoch count, batching, client concurrency and RF equal when comparing allocators.
`MOONCAKE_DFS_DEFERRED_FREE_SECONDS=0` can expose offset reuse in a short experiment, but differs from the default 30-second deferred free.
Chunk alignment can increase reserved space substantially: shard allocation reserves aligned size plus padding and allocator quantization.
A bucket allocator can still cause COW during partial chunk extension and overwrite.

Confirm USRBIO write and read submissions independently; successful `get()` alone cannot distinguish a memory hit from 3FS I/O.
`close()` may leave a memory replica visible temporarily in the tested revision, so this benchmark allocates and explicitly unmounts its own segments.
Every put and get result is checked, restored buffers are zeroed before reading, and all old keys are probed after invalidation.

`events.jsonl` records individual calls and phase envelopes.
Use phase durations for throughput with concurrent clients because summed request durations overlap.
Peak RSS is process peak RSS; process CPU includes setup, checksum validation and teardown.
Per-call CPU durations overlap with other threads.
Size the experimental 64 MiB staging pool for the live in-flight workload before production use.

Use fio's official USRBIO engine to isolate overwrite geometry with the same chunk size and request bytes.
It omits Store metadata, allocator placement, duplicate suppression, staging copies and invalidation.
Neither fio nor this CPU benchmark demonstrates an end-to-end Agent RL speedup or NAND write amplification.

The client and its FD cache persist across policy epochs, matching the worker lifecycle.
Only the explicitly allocated memory segment is unmounted to force DFS restoration, after all submitted operations complete.
CUDA fixtures synchronize before submission and after native completion; CPU checksums are outside individual I/O phase timings.

The client's `--local-buffer-bytes` supplies temporary contiguous storage for
DFS multi-buffer reads and is distinct from FileStorage's offload RPC arena
(`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`).
A 4 MiB payload with four keys per batch and four concurrent batches can need
64 MiB of restore buffers: the initial 16 MiB fixture fails with error `-200`.
Use `--local-buffer-bytes 134217728` for those scenarios, keep it equal between
allocator variants and repetitions, and disclose the larger buffer capacity.
The raw `staging_pool_bytes` summary field records the offload RPC arena setting,
not the independent GPU D2H pinned-buffer pool.
