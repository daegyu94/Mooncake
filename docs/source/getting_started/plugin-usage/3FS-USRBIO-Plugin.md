# Mooncake HF3FS USRBIO Adapter (Experimental)

```{warning}
**Work in progress / experimental.** Descriptor-based DFS and its HF3FS (3FS
USRBIO) adapter are under development and are not production-ready. They are
not covered by Mooncake Store's general fault-tolerance, HA continuity,
durability, or multi-tenant guarantees. Behavior, build flags, and
configuration may change without notice. Use only for evaluation and testing.
```

This adapter implements the HF3FS native USRBIO data plane for Mooncake Store's
descriptor-based DFS replicas. The master allocates ranges in shared shard
files, and clients use USRBIO to access the ranges described by the replica
metadata.

The adapter is not enabled by the legacy `--root_fs_dir` option. It also does
not automatically fall back to POSIX I/O; select
`MOONCAKE_DFS_FS_ADAPTER=posix` explicitly when POSIX behavior is required.

## Prerequisites

### 1. HF3FS installation

- Build and install [3FS](https://github.com/deepseek-ai/3FS/)
- Required library: `libhf3fs_api_shared.so` (Default location: `3FS_PATH/build/src/lib/api`)
  → Install to: `/usr/lib/`
- Required header: `hf3fs_usrbio.h` (Default location: `3FS_PATH/src/lib/api`)
  → Install to: `/usr/include/`

### 2. Mooncake build

Enable HF3FS support during CMake configuration:

```bash
cmake -DUSE_3FS=ON ...
```

Then build and install Mooncake as usual.

## Usage

### Master

Enable descriptor-based DFS and point it at a directory on the shared HF3FS
mount:

```bash
export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_ROOT_DIR=/mnt/3fs/mooncake
export MOONCAKE_DFS_FS_ADAPTER=hf3fs
export MOONCAKE_DFS_SHARD_COUNT=64
export MOONCAKE_DFS_SHARD_CAPACITY=4294967296
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=true

./build/mooncake-store/src/mooncake_master [other master arguments]
```

The master creates and preallocates the configured shard files during startup.
Ensure the mount is available, writable, and has enough capacity before
starting the process.

### Clients

Every client that may read or write DFS replicas must use the same root,
adapter, shard count, shard capacity, and alignment. The DFS root must be an
absolute path and use the same path string in every process. For the standalone
store service:

```bash
export MOONCAKE_OFFLOAD_ENABLED=true
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/data/file_storage
export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_DFS_ROOT_DIR=/mnt/3fs/mooncake
export MOONCAKE_DFS_FS_ADAPTER=hf3fs
export MOONCAKE_DFS_SHARD_COUNT=64
export MOONCAKE_DFS_SHARD_CAPACITY=4294967296
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=true

python -m mooncake.mooncake_store_service
```

`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` must already be an absolute, writable,
non-symlink directory. DFS shard data is stored under
`MOONCAKE_DFS_ROOT_DIR`; the separate FileStorage path is still validated
during client initialization.

For the complete configuration reference, request example, synchronous write
semantics, and current recovery limitations, see the {ref}`DFS deployment
documentation <dfs-storage>`.

### Descriptor I/O batching and concurrency

Descriptor-based DFS reads and writes prepare a bounded window of native I/O before submitting it. Each request owns a separate region of the thread's shared IOV; the adapter drains every prepared request before returning, including after an error. Completion indices and exact byte counts are checked, and per-object results retain the input order.

Configure these limits on every DFS client before initialization:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MOONCAKE_DFS_IO_BATCH_SIZE` | `16` | Maximum requests per window, capped by the thread's ring capacity. |
| `MOONCAKE_DFS_IO_BATCH_BYTES` | `33554432` | Maximum shared IOV bytes per window, including alignment padding; minimum `4096`, capped by the existing IOV size. |

Set the request limit to `1` to use sequential submission. Calls do not accumulate work across batches or add a batching delay. The existing 32 MiB thread IOV allocation is unchanged, and file-level `ReadFile`/`WriteFile` operations do not use this batching path.

Within one backend instance, reads may run concurrently with a write to a disjoint allocation range in the same shard. Overlapping reads are allowed; a write excludes overlapping reads until its I/O finishes. Writes to the same shard remain serialized to protect filesystem write metadata beyond individual byte ranges. Shard leases are acquired in ascending shard order, and earlier conflicting waiters are respected. This is process-local coordination: the allocator and descriptor protocol still govern cross-process ownership and late writers.

The native API provides no cancellation guarantee for already prepared I/O. A failed submit notification still requires completion draining; if the native completion path stops permanently, the call can remain blocked. A malformed completion fences the thread resource from subsequent descriptor submissions.
