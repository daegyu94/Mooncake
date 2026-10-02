# Agent RL DFS workstation results

Source pin: `e88aacf20cd83461ac4e6ff18a8b50a1e2b3f349`.
Runtime wheel: `0.3.12.post1`, master stamp 0.3.13/e88aacf, matching vendor USRBIO library; exact binary SHA-256 values are recorded separately.
The wheel/native library were reused in an isolated environment; no native optimization build or storage service replacement is implied.

Sixty primary native runs cover CPU payloads 64 KiB, 448 KiB, 512 KiB, 1 MiB and 4 MiB, concurrency 1/4, shard/bucket, selected 1 MiB alignment conditions, an offload RPC arena A/B and CUDA staging with 448 KiB/4 MiB fixtures.
Each condition has two repeats, 16 keys, four epochs, batch four, 56 slices, 1 GiB nominal DFS capacity, memory segments 256 MiB, RF1.
They use persistent clients, explicit memory unmounts, zeroed DFS destinations, actual USRBIO read traces, SHA-256 checks and drained old-key invalidation.
Deferred free is set to zero for short offset-reuse stress, unlike trainer runs' default 30 seconds.

For 448 KiB CPU payloads at concurrency 1:

| Allocator | Put MiB/s mean | Extra COW read / logical write | Data pwrite / logical write | FD registrations |
|---|---:|---:|---:|---:|
| shard, 4 KiB alignment | 24.780 | 2.059 | 2.427 | 4 |
| bucket | 20.720 | 0.321 | 1.357 | 128 |
| shard, 1 MiB alignment | 31.154 | 0 | 1.000 | 4 |

Bucket does not eliminate COW and is slower in this condition.
At 64 KiB/concurrency 4 it increased amplification versus shard.
Chunk alignment reduces I/O but the current allocator reserves aligned size plus alignment padding and size-class rounding: useful 448 KiB KV capacity is approximately 224 MiB within 1 GiB nominal storage.
This tiny workload did not exercise eviction/hit-rate penalties.

The FileStorage offload RPC arena A/B reduces 1.25 GiB to 64 MiB.
With identical native KV operations, average peak RSS is approximately 1,413,000 to 169,000 KiB and process CPU about 0.639 to 0.306 seconds.
I/O amplification is unchanged; put/get throughput is slightly lower, so this is a memory/initialization saving, not an I/O-throughput improvement.
This arena is separate from the client temporary restore buffer, GPU D2H pinned pool and USRBIO buffers. 4 MiB/batch4/concurrency4 failed with client buffer16 MiB (-200); all matched 4 MiB primary cases use128 MiB and exclude the failed/legacy pilot.

The prior `dedup_suppressed_put_bytes` summary field at `4466e4d` is unmeasured and removed at `08f0291`; prior raw values must be ignored.
Earlier memory-get destinations were not cleared, so separate zeroed-destination same-key policy tests provide stronger memory-tier copy correctness evidence.
These fixtures are not trainer modes or E2E rollout benchmarks.
Real veRL tests observed actual USRBIO puts but no external restore GET.
Full condition and raw-sample tables are in `agent-rl-dfs-results.csv` and the companion veRL experiment branch `experiments/agent-rl-3fs-cost`.
