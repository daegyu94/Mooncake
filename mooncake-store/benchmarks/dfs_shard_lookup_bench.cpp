// Backend CPU benchmark: real shard initialization, no storage transfer cost.
// Compare the same executable source against both backend revisions.
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <barrier>
#include <chrono>
#include <time.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <thread>

#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/posix_fs_adapter.h"

DEFINE_int32(batch, 64, "Requests per batch");
DEFINE_int32(shards, 1, "Distinct shards, interleaved within each batch");
DEFINE_int32(threads, 1, "Concurrent callers");
DEFINE_int32(iterations, 10000, "Batches per caller");

namespace {
class NoTransferAdapter : public mooncake::PosixFsAdapter {
   public:
    bool SupportsBatchIo() const override { return true; }
    bool SupportsFdReopen() const override { return false; }
    tl::expected<size_t, mooncake::ErrorCode> ReadAt(int, iovec* iov, int count,
                                                     int64_t) override {
        size_t size = 0;
        for (int i = 0; i < count; ++i) size += iov[i].iov_len;
        return size;
    }
};

struct TestDirectory {
    std::string path;
    TestDirectory() {
        char pattern[] = "/tmp/dfs_shard_lookup_XXXXXX";
        auto* dir = ::mkdtemp(pattern);
        CHECK(dir);
        path = dir;
    }
    ~TestDirectory() { std::filesystem::remove_all(path); }
};
}  // namespace

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    CHECK_GT(FLAGS_batch, 0);
    CHECK_GT(FLAGS_shards, 0);
    CHECK_LE(FLAGS_shards, FLAGS_batch);
    CHECK_GT(FLAGS_threads, 0);
    CHECK_GT(FLAGS_iterations, 0);
    TestDirectory directory;
    mooncake::DistributedStorageConfig config;
    config.fsdir = directory.path;
    config.shard_capacity = 4096;
    config.alignment = 4096;
    auto adapter = std::make_unique<NoTransferAdapter>();
    CHECK(adapter->Init(directory.path));
    std::vector<std::string> paths;
    for (int i = 0; i < FLAGS_shards; ++i) {
        paths.push_back(directory.path + "/dfs_shard_" + (i < 10 ? "0" : "") +
                        std::to_string(i) + ".data");
        CHECK(adapter->PreallocateFile(paths.back(), config.shard_capacity));
    }
    mooncake::FileStorageConfig file_config;
    mooncake::DistributedStorageBackend backend(file_config, config,
                                                std::move(adapter));
    CHECK(backend.Init());
    std::barrier ready(FLAGS_threads + 1), start(FLAGS_threads + 1),
        finish(FLAGS_threads + 1);
    std::vector<double> cpu_times(FLAGS_threads);
    std::vector<std::thread> workers;
    for (int worker = 0; worker < FLAGS_threads; ++worker) {
        workers.emplace_back([&, worker] {
            char output[4]{};
            std::vector<mooncake::DfsReadRequest> requests;
            for (int i = 0; i < FLAGS_batch; ++i) {
                int shard = i % FLAGS_shards;
                requests.push_back({"key",
                                    {paths[shard], 0, 4, 4096, shard},
                                    {{output, sizeof(output)}}});
            }
            for (const auto& result : backend.BatchRead(requests))
                CHECK(result);
            ready.arrive_and_wait();
            start.arrive_and_wait();
            timespec cpu_start{}, cpu_end{};
            CHECK_EQ(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start), 0);
            for (int i = 0; i < FLAGS_iterations; ++i) {
                for (const auto& result : backend.BatchRead(requests))
                    CHECK(result);
            }
            CHECK_EQ(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end), 0);
            cpu_times[worker] =
                double(cpu_end.tv_sec - cpu_start.tv_sec) +
                double(cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e9;
            finish.arrive_and_wait();
        });
    }
    ready.arrive_and_wait();
    const auto wall_start = std::chrono::steady_clock::now();
    start.arrive_and_wait();
    finish.arrive_and_wait();
    const auto elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - wall_start)
                             .count();
    for (auto& worker : workers) worker.join();
    double cpu = 0;
    for (double time : cpu_times) cpu += time;
    const double batches = double(FLAGS_iterations) * FLAGS_threads;
    std::cout << "batch=" << FLAGS_batch << " shards=" << FLAGS_shards
              << " threads=" << FLAGS_threads
              << " ns_per_batch=" << elapsed * 1e9 / batches
              << " cpu_ns_per_batch=" << cpu * 1e9 / batches
              << " requests_per_second=" << batches * FLAGS_batch / elapsed
              << '\n';
}
