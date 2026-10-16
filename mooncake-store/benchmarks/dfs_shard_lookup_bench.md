# DFS shard lookup CPU benchmark

`dfs_shard_lookup_bench` measures the descriptor-based `DistributedStorageBackend::BatchRead()` path using a batch-capable POSIX adapter that reports successful reads without transferring payloads. Shards are real, prepared sparse files and are validated and opened during warmup. This isolates backend request preparation, shard resolution, allocation and locking costs; its request rate is not storage or end-to-end GET throughput.

## Running the comparison

Build with `BUILD_BENCHMARK=ON`, then run the same benchmark source against the base and candidate backend revisions with the same Release configuration and dependencies. The base does not contain this target, so copy the benchmark source and its CMake target into a separate base checkout before building. Keep timing runs separate from instrumented runs and other builds.

```bash
cmake --build build --target dfs_shard_lookup_bench
build/mooncake-store/benchmarks/dfs_shard_lookup_bench \
  --batch=64 --shards=1 --threads=1 --iterations=10000
```

Requests interleave `shards` distinct shard indices, including nonadjacent reuse. Sweep batch/shards/callers through `(1,1,1)`, `(16,1,1)`, `(64,1,1)`, `(64,4,1)`, `(64,64,1)`, `(64,1,4)`, `(64,4,4)` and `(64,64,4)`. Use at least 200,000 iterations for the short batch-1 control. Run five repetitions with alternating before/after order and fixed CPU affinity. `ns_per_batch` is elapsed wall time divided by all completed batches; with concurrent callers it is an aggregate service interval, not individual request latency. `cpu_ns_per_batch` sums caller thread CPU clocks and divides by completed batches. The benchmark warms every caller before starting the timers and checks every result.

## Implementation and invariants

Before this change, every validated operation in `ExecuteShardBatchIo()` calls `GetOrOpenShard()`. A successful warm lookup normalizes the descriptor/root paths, parses the shard filename, acquires the global shard-cache mutex, looks up the shard, acquires its initialization mutex and resolves its fd. The batch adapter receives the resulting requests synchronously. Current upstream HF3FS uses the default sequential adapter batch methods, so per-request HF3FS resource lookup and USRBIO preparation remain outside this optimization.

The candidate memoizes successful `(shard_idx, raw file_path)` resolutions inside each `ExecuteShardBatchIo()` call. The small map holds one raw-path view and fd per encountered shard index; a different path always returns to the original validator and replaces that entry only after success. Failures are not cached, so a subsequent request can retry an unsuccessful open within the same batch. Range, buffer and size validation still run separately for every request. Request order, offsets, error mapping and short-transfer behavior are unchanged. The helper is shared by batch GET and PUT; POSIX scalar I/O, bucket I/O, public APIs and adapter implementations are unchanged.

The global mutex still protects `shard_files_` lookup/insertion. Each shard mutex still protects initialization and non-batch positional I/O/reopen. Entries are never erased while the backend runs. Batch-capable adapters already use stable fds without holding the shard mutex during I/O and never execute backend reopen; the memoization relies on that existing invariant. Concurrent batches own separate maps, and dynamic shard publication still goes through the original lazy-open synchronization.

No persistent cache or synchronization primitive is added. Temporary memory is O(U) for U distinct shard indices, including unordered-map nodes and buckets; path strings are not copied. Single-operation batches allocate no map entries. Failed resolutions or alternating raw aliases of the same index can cause additional lookups. Initialization must complete before I/O, and destruction must wait for active calls to finish, as required by the existing unsynchronized `initialized_` and destructor implementation. This change does not add concurrent initialization/shutdown support or new same-object read/write atomicity guarantees.

## Measured results

Measurements were taken on 2026-10-16 with GCC 13.3, a Release `-O3` build, Linux x86-64 and an AMD Ryzen Threadripper 9960X (24 cores/48 threads). Calls were pinned to logical CPUs 2–5. Base backend: upstream `049d887eefbf5ad9ffb6bd8c01d56f3de0522826`. Before and after used the same final benchmark object and dependency libraries, with only the backend object replaced for the baseline. Values below are medians of five runs; CPU frequency was not fixed.

| Batch | Shards | Callers | CPU µs/batch before → after | CPU change | Wall µs/batch before → after |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0.657 → 0.662 | +0.8% | 0.657 → 0.663 |
| 16 | 1 | 1 | 8.870 → 1.201 | -86.5% | 8.872 → 1.203 |
| 64 | 1 | 1 | 35.933 → 3.547 | -90.1% | 35.940 → 3.568 |
| 64 | 4 | 1 | 36.010 → 5.144 | -85.7% | 36.028 → 5.203 |
| 64 | 64 | 1 | 35.863 → 38.485 | +7.3% | 35.919 → 38.490 |
| 64 | 1 | 4 | 37.962 → 3.182 | -91.6% | 10.426 → 1.352 |
| 64 | 4 | 4 | 35.065 → 4.710 | -86.6% | 9.187 → 1.280 |
| 64 | 64 | 4 | 34.722 → 37.097 | +6.8% | 9.596 → 10.012 |

At batch 64 with one shard and one caller, backend request rate increased from 1.78 to 17.94 million requests/s. With four shards it increased from 1.78 to 12.30 million requests/s. Unique-shard batches add map overhead without avoiding any resolutions: CPU cost rose by 7.3% with one caller and 6.8% with four callers. The longer batch-1 control rose by 0.8%. These controls show the tradeoff rather than establishing absence of regressions.

Successful warm shard resolutions and mutex acquisitions were independently counted in temporary instrumented binaries, outside the timing comparison. A counter was inserted at `GetOrOpenShard()` entry, and GNU ld `--wrap=pthread_mutex_lock` counted mutex acquisitions. Thread-local measurement gates excluded warmup and barriers. For 100 batches per caller, observed totals were:

| Batch | Shards | Callers | Resolve calls before → after | Mutex acquisitions before → after |
| --- | --- | --- | --- | --- |
| 64 | 1 | 1 | 6,400 → 100 | 12,800 → 200 |
| 64 | 4 | 1 | 6,400 → 400 | 12,800 → 800 |
| 64 | 64 | 1 | 6,400 → 6,400 | 12,800 → 12,800 |
| 64 | 1 | 4 | 25,600 → 400 | 51,200 → 800 |

For ordinary successful descriptors with one raw path per shard, N requests addressing U shards therefore resolve fds N → U times and acquire backend mutexes 2N → 2U times. These are backend resolution counts, not OS file-open counts: the existing persistent shard cache already avoids repeated successful opens. Lock waiting time and hardware CPU profiles were not measured; the four-caller cases exercise contention without isolating wait duration.

## Validation and scope

Release `dfs_posix_test` passed all 60 tests, including seven added batch-specific tests. These cover nonadjacent mixed-shard reads/writes, raw aliases and invalid identities, a different prepared filename for the same shard index, GET/PUT open retry within a batch, request-level failures/short transfers and concurrent lazy-open access across five shards, including a newly published shard. Existing tests cover POSIX reopen success/failure, failed initialization retries, independent-shard progress and allocator expansion/rollback. Concurrent data tests use disjoint offsets.

Release `dfs_immutable_bucket_backend_test` passed all seven tests. Native `dfs_hf3fs_test` passed all four tests without skips against a local HF3FS mount, including same-shard concurrent batches and bounded bucket submissions. An initial native run failed to create its test directory because the mount root was owned by root; using an isolated directory below an existing user-owned directory resolved the environment failure.

All 60 POSIX tests also passed with the backend and test translation units compiled with ASan/UBSan; leak detection was disabled and dependency archives were not sanitizer-instrumented. ThreadSanitizer initially aborted before tests with `unexpected memory mapping`; disabling ASLR for the test process with `setarch x86_64 -R` allowed all 60 POSIX tests to pass, including the batch/concurrency tests. Only the backend and test translation units were instrumented; dependency archives were reused. This changes no global host setting.

Independent agents reviewed the plan and final code with no blocking findings. Scoped pre-commit hooks passed except for pre-existing `cmake-format` differences in unchanged sections of the benchmark CMake file. The same tool fails on the base, and its formatted unchanged prefix is identical on base and candidate. Those unrelated edits are excluded; the added target itself matches the formatter.

This is incremental to [PR #4498](https://github.com/kvcache-ai/Mooncake/pull/4498), which introduced opt-in adapter batching and bucket-handle reuse, and does not duplicate [PR #4510](https://github.com/kvcache-ai/Mooncake/pull/4510) or [RFC #4497](https://github.com/kvcache-ai/Mooncake/issues/4497), which cover HF3FS USRBIO ring batching. Searches of upstream issues/PRs for DFS batching, shard lookup and the exact function name found no equivalent shard-resolution memoization at the time of this work. The pending #4510 changes adapter/resource-manager code rather than this backend helper; it has not been integration-tested together with this patch.

The benchmark intentionally exposes map overhead when every request addresses a different shard; report that control alongside repeated-shard gains. Production GET tail latency, storage throughput, allocator memory peaks and lock wait-time profiles were not measured. Results isolate backend CPU work and must not be extrapolated to network/storage performance.
