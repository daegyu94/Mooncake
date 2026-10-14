#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "client_service.h"
#include "common/client_buffer_allocation.h"
#include "embedded_master.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/hf3fs_adapter.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
double Seconds(const timeval& value) {
    return value.tv_sec + value.tv_usec / 1e6;
}
}  // namespace

// Real Client/Master, local MEMORY and native USRBIO. Both variants use the
// fenced protocol; only cross-tier scheduling differs. This is not a trainer
// benchmark or a remote RDMA/NoF bandwidth measurement.
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: parallel_dfs_put_bench ROOT serial|parallel "
                     "ITERATIONS BATCH BYTES host|gpu\n";
        return 2;
    }
    using namespace mooncake;
    const std::filesystem::path root = argv[1];
    const bool parallel = std::string(argv[2]) == "parallel";
    const size_t iterations = std::stoull(argv[3]);
    const size_t batch = std::stoull(argv[4]);
    const size_t bytes = std::stoull(argv[5]);
    const bool gpu = std::string(argv[6]) == "gpu";
    Require(root.is_absolute(), "namespace must be absolute");
    Require(
        std::string(argv[2]) == "serial" || std::string(argv[2]) == "parallel",
        "invalid mode");
    Require(std::string(argv[6]) == "host" || std::string(argv[6]) == "gpu",
            "invalid source");
    Require(iterations > 0 && iterations <= 1000000 && batch > 0 &&
                batch <= 64 && bytes > 0 && bytes <= 16 * 1024 * 1024,
            "invalid workload");
    Require(!std::filesystem::exists(root), "namespace already exists");
    std::filesystem::create_directories(root);
    setenv("MOONCAKE_ENABLE_DFS", "1", 1);
    setenv("MC_STORE_DFS_WRITE_FENCING", "1", 1);
    setenv("MC_STORE_PARALLEL_DFS_PUT", parallel ? "1" : "0", 1);
    setenv("MC_STORE_MEMCPY", "1", 1);
    setenv("MOONCAKE_DFS_ROOT_DIR", root.c_str(), 1);
    setenv("MOONCAKE_DFS_FS_ADAPTER", "hf3fs", 1);
    setenv("MOONCAKE_DFS_SHARD_COUNT", "1", 1);
    setenv("MOONCAKE_DFS_SHARD_CAPACITY", "268435456", 1);
    setenv("MOONCAKE_DFS_ALIGNMENT", "4096", 1);
    setenv("MOONCAKE_DFS_EVICTION_ENABLED", "0", 1);
    setenv("MOONCAKE_DFS_DEFERRED_FREE_SECONDS", "0", 1);
    setenv("MOONCAKE_DFS_SINGLE_TENANT", "true", 1);
    EmbeddedMaster master;
    Require(master.Start(InProcMasterConfigBuilder().build()),
            "master start failed");
    auto created = Client::Create("127.0.0.1:19001", "P2PHANDSHAKE", "tcp",
                                  std::nullopt, master.master_address());
    Require(created.has_value(), "client creation failed");
    auto client = *created;
    const size_t segment_bytes =
        std::max<size_t>(64 * 1024 * 1024, batch * bytes * 2);
    void* segment = allocate_buffer_allocator_memory(segment_bytes);
    Require(segment != nullptr, "memory allocation failed");
    Require(client->MountSegment(segment, segment_bytes, "tcp").has_value(),
            "mount failed");
    FileStorageConfig file;
    file.storage_backend_type = StorageBackendType::kDistributed;
    file.storage_filepath = root.string();
    DistributedStorageConfig storage;
    storage.fsdir = root.string();
    storage.fs_adapter_type = "hf3fs";
    storage.shard_count = 1;
    storage.shard_capacity = 268435456;
    auto backend = std::make_shared<DistributedStorageBackend>(
        file, storage, std::make_unique<Hf3fsAdapter>());
    Require(backend->Init().has_value(), "backend init failed");
    client->SetDfsStorageBackend(backend);
    std::vector<char> value(bytes, 'V');
    void* source = value.data();
#ifdef USE_CUDA
    if (gpu) {
        Require(cudaMalloc(&source, bytes) == cudaSuccess,
                "CUDA allocation failed");
        Require(cudaMemcpy(source, value.data(), bytes,
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "CUDA preparation failed");
    }
#else
    Require(!gpu, "GPU workload requires USE_CUDA");
#endif
    ReplicateConfig replicas;
    replicas.dfs_replica_num = 1;
    std::vector<std::vector<Slice>> slices(batch, {{source, bytes}});
    std::vector<double> latencies;
    latencies.reserve(iterations);
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    const auto loop_start = Clock::now();
    for (size_t run = 0; run < iterations; ++run) {
        std::vector<std::string> keys;
        for (size_t i = 0; i < batch; ++i)
            keys.push_back("bench_" + std::to_string(run) + "_" +
                           std::to_string(i));
        const auto start = Clock::now();
        const auto results = client->BatchPut(keys, slices, replicas);
        latencies.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start)
                .count());
        Require(results.size() == batch, "invalid put result count");
        for (const auto& result : results)
            Require(result.has_value(), "put failed");
        if (run == 0) {
            auto query = client->Query(keys[0]);
            Require(query.has_value(), "query failed");
            const auto dfs = std::find_if(
                query->replicas.begin(), query->replicas.end(),
                [](const auto& replica) { return replica.is_dfs_replica(); });
            Require(dfs != query->replicas.end(), "DFS replica missing");
            std::vector<char> read(bytes);
            const auto reads =
                backend->BatchRead({{keys[0],
                                     dfs->get_dfs_descriptor(),
                                     {{read.data(), read.size()}}}});
            Require(reads.size() == 1 && reads[0].has_value() && read == value,
                    "DFS verification failed");
        }
        const auto removed = client->BatchRemove(keys, true);
        Require(removed.size() == batch, "invalid remove result count");
        for (const auto& result : removed)
            Require(result.has_value(), "remove failed");
    }
    const double loop_seconds =
        std::chrono::duration<double>(Clock::now() - loop_start).count();
    getrusage(RUSAGE_SELF, &after);
    double total_us = 0;
    for (const auto us : latencies) total_us += us;
    std::sort(latencies.begin(), latencies.end());
    auto percentile = [&](double fraction) {
        return latencies[static_cast<size_t>((latencies.size() - 1) *
                                             fraction)];
    };
    const double gib = static_cast<double>(iterations) * batch * bytes /
                       (1024.0 * 1024 * 1024);
    std::cout << "{\"parallel\":" << (parallel ? "true" : "false")
              << ",\"gpu\":" << (gpu ? "true" : "false")
              << ",\"iterations\":" << iterations << ",\"batch\":" << batch
              << ",\"bytes\":" << bytes << ",\"p50_us\":" << percentile(0.5)
              << ",\"p95_us\":" << percentile(0.95)
              << ",\"p99_us\":" << percentile(0.99)
              << ",\"put_gib_s\":" << gib / (total_us / 1e6)
              << ",\"loop_gib_s\":" << gib / loop_seconds << ",\"cpu_s\":"
              << Seconds(after.ru_utime) + Seconds(after.ru_stime) -
                     Seconds(before.ru_utime) - Seconds(before.ru_stime)
              << ",\"max_rss_kib\":" << after.ru_maxrss << "}\n";
#ifdef USE_CUDA
    if (gpu) cudaFree(source);
#endif
    Require(client->UnmountSegment(segment, segment_bytes).has_value(),
            "unmount failed");
    client.reset();
    created.reset();
    backend.reset();
    master.Stop();
    std::free(segment);
    // This harness owns the unique namespace and removes it only after all
    // successful synchronous calls and all component owners have stopped.
    std::filesystem::remove_all(root);
}
