#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <array>

#include <unistd.h>

#include "client_service.h"
#include "../src/config/client_object_checksum_config.h"
#include "storage/distributed/shard_allocator.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/posix_fs_adapter.h"
#ifdef USE_3FS
#include "storage/distributed/hf3fs_adapter.h"
#endif
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif
#include "test_server_helpers.h"
#include "master_service/master_service_test_peer.h"
#include "common/client_buffer_allocation.h"

namespace mooncake::test {

class FailingPosixFsAdapter : public PosixFsAdapter {
   public:
    int WriteCalls() const { return write_calls_.load(); }
    void FailWriteCall(int call) { fail_write_call_.store(call); }
    void BlockWriteCall(int call) {
        std::lock_guard lock(block_mutex_);
        block_call_ = call;
        blocked_ = false;
        released_ = false;
    }
    bool WaitForBlockedWrite() {
        std::unique_lock lock(block_mutex_);
        return block_cv_.wait_for(lock, std::chrono::seconds(2),
                                  [this] { return blocked_; });
    }
    void ReleaseWrite() {
        std::lock_guard lock(block_mutex_);
        released_ = true;
        block_cv_.notify_all();
    }
    int64_t BlockedOffset() const { return blocked_offset_.load(); }

    tl::expected<size_t, ErrorCode> WriteAt(int fd, const iovec* iov,
                                            int iovcnt,
                                            int64_t offset) override {
        const int call = ++write_calls_;
        {
            std::unique_lock lock(block_mutex_);
            if (call == block_call_) {
                blocked_offset_ = offset;
                blocked_ = true;
                block_cv_.notify_all();
                block_cv_.wait(lock, [this] { return released_; });
            }
        }
        const int fail = fail_write_call_.load();
        // Model a genuine, non-recoverable write failure: once the armed call
        // is reached, keep failing. The backend retries POSIX write failures
        // once after an fd reopen (SupportsFdReopen), so a single-call failure
        // would be masked by the retry; a persistent failure correctly
        // surfaces to the caller.
        if (fail > 0 && call >= fail) {
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
        return PosixFsAdapter::WriteAt(fd, iov, iovcnt, offset);
    }

   private:
    std::atomic<int> write_calls_{0};
    std::atomic<int> fail_write_call_{-1};
    std::mutex block_mutex_;
    std::condition_variable block_cv_;
    int block_call_ = -1;
    bool blocked_ = false;
    bool released_ = true;
    std::atomic<int64_t> blocked_offset_{-1};
};

// A host-backed device double exercises the production memcpy worker and
// DFS staging paths. It supplies ordering barriers, not CUDA correctness.
class BlockingWriteDevice final : public device::AcceleratorDevice {
   public:
    device::AcceleratorVendor Vendor() const override {
        return device::AcceleratorVendor::kIntel;
    }
    bool Available(bool = false) const override { return true; }
    device::PointerInfo QueryPointer(const void* ptr) const override {
        return {ptr == source_.load() ? device::MemoryKind::kDevice
                                      : device::MemoryKind::kHost,
                0};
    }
    int32_t CurrentDeviceId() const override { return 0; }
    void SetContext(int32_t) const override {}
    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        return {new char[size], size,
                [](void* p) { delete[] static_cast<char*>(p); }};
    }
    bool Copy(void* dst, const void* src, size_t size,
              device::CopyDirection) const override {
        std::unique_lock lock(mutex_);
        const auto address = reinterpret_cast<uintptr_t>(dst);
        const bool memory =
            address >= memory_begin_ && address - memory_begin_ < memory_size_;
        if (memory) {
            memory_started_ = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return released_; });
            if (fail_memory_) return false;
        } else {
            staging_started_ = true;
            cv_.notify_all();
            if (fail_staging_) return false;
        }
        lock.unlock();
        std::memcpy(dst, src, size);
        return true;
    }
    void Arm(void* source, void* memory, size_t memory_size,
             bool fail_memory = false, bool fail_staging = false) {
        std::lock_guard lock(mutex_);
        memory_begin_ = reinterpret_cast<uintptr_t>(memory);
        memory_size_ = memory_size;
        fail_memory_ = fail_memory;
        fail_staging_ = fail_staging;
        released_ = memory_started_ = staging_started_ = false;
        source_ = source;
    }
    bool WaitForMemory() const {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2),
                            [this] { return memory_started_; });
    }
    bool WaitForStaging() const {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::milliseconds(200),
                            [this] { return staging_started_; });
    }
    void Release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
    void Disarm() { source_ = nullptr; }

   private:
    std::atomic<void*> source_{nullptr};
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    uintptr_t memory_begin_ = 0;
    size_t memory_size_ = 0;
    bool fail_memory_ = false;
    bool fail_staging_ = false;
    mutable bool memory_started_ = false;
    mutable bool staging_started_ = false;
    bool released_ = true;
};

BlockingWriteDevice blocking_device;
const device::AcceleratorDeviceRegistrar blocking_registration(blocking_device);

class DfsSyncClientTest : public ::testing::Test {
   protected:
    virtual std::filesystem::path TestRoot() const {
        return std::filesystem::temp_directory_path();
    }
    virtual const char* AdapterName() const { return "posix"; }
    virtual std::unique_ptr<FileSystemAdapter> CreateAdapter() {
        return std::make_unique<FailingPosixFsAdapter>();
    }
    void SetUp() override {
        root_ = (TestRoot() / ("dfs_sync_client_" + std::to_string(::getpid()) +
                               "_" + std::to_string(++next_root_)))
                    .string();
        std::filesystem::create_directories(root_);

        SetEnv("MOONCAKE_ENABLE_DFS", "1");
        SetEnv("MOONCAKE_DFS_FS_ADAPTER", AdapterName());
        SetEnv("MOONCAKE_DFS_ROOT_DIR", root_);
        SetEnv("MOONCAKE_DFS_SHARD_COUNT", "2");
        SetEnv("MOONCAKE_DFS_SHARD_CAPACITY", "16777216");
        SetEnv("MOONCAKE_DFS_ALIGNMENT", "4096");
        SetEnv("MOONCAKE_DFS_EVICTION_ENABLED", "0");
        SetEnv("MOONCAKE_DFS_DEFERRED_FREE_SECONDS", "0");
        SetEnv("MOONCAKE_DFS_SINGLE_TENANT", "true");

        ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder().build()));
        writer_ = CreateClient("127.0.0.1:18101");
        provider_ = CreateClient("127.0.0.1:18102");
        ASSERT_NE(writer_, nullptr);
        ASSERT_NE(provider_, nullptr);

        segment_size_ = 16 * 1024 * 1024;
        segment_ = allocate_buffer_allocator_memory(segment_size_);
        ASSERT_NE(segment_, nullptr);
        ASSERT_TRUE(provider_->MountSegment(segment_, segment_size_, "tcp")
                        .has_value());

        FileStorageConfig file_config;
        file_config.storage_backend_type = StorageBackendType::kDistributed;
        file_config.storage_filepath = root_;

        DistributedStorageConfig distributed_config;
        distributed_config.fsdir = root_;
        distributed_config.fs_adapter_type = AdapterName();
        distributed_config.shard_count = 2;
        distributed_config.shard_capacity = 16 * 1024 * 1024;
        distributed_config.alignment = 4096;

        auto adapter = CreateAdapter();
        adapter_ = dynamic_cast<FailingPosixFsAdapter*>(adapter.get());
        backend_ = std::make_shared<DistributedStorageBackend>(
            file_config, distributed_config, std::move(adapter));
        ASSERT_TRUE(backend_->Init().has_value());
        writer_->SetDfsStorageBackend(backend_);
    }

    void TearDown() override {
        if (provider_ && segment_) {
            (void)provider_->UnmountSegment(segment_, segment_size_);
        }
        writer_.reset();
        provider_.reset();
        backend_.reset();
        master_.Stop();
        if (segment_) {
            std::free(segment_);
            segment_ = nullptr;
        }
        RestoreEnv();
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    std::shared_ptr<Client> CreateClient(const std::string& hostname) {
        auto client = Client::Create(hostname, "P2PHANDSHAKE", "tcp",
                                     std::nullopt, master_.master_address());
        return client ? *client : nullptr;
    }

    ReplicateConfig DfsConfig() const {
        ReplicateConfig config;
        config.replica_num = 1;
        config.dfs_replica_num = 1;
        return config;
    }

    tl::expected<QueryResult, ErrorCode> QueryDfsOnly(const std::string& key) {
        auto query = writer_->Query(key);
        if (!query) {
            return tl::make_unexpected(query.error());
        }
        std::vector<Replica::Descriptor> replicas;
        for (const auto& replica : query->replicas) {
            if (replica.is_dfs_replica()) {
                replicas.push_back(replica);
            }
        }
        if (replicas.empty()) {
            return tl::make_unexpected(ErrorCode::INVALID_REPLICA);
        }
        return QueryResult(std::move(replicas), query->lease_timeout,
                           query->object_checksum);
    }

    void ExpectDfsValue(const std::string& key, const std::string& expected) {
        auto query = QueryDfsOnly(key);
        ASSERT_TRUE(query.has_value());
        last_read_lease_deadline_ = query->lease_timeout;
        std::vector<char> output(expected.size());
        const auto& descriptor = query->replicas[0].get_dfs_descriptor();
        auto results = backend_->BatchRead(
            {{key, descriptor, {{output.data(), output.size()}}}});
        ASSERT_EQ(results.size(), 1);
        ASSERT_TRUE(results[0].has_value());
        EXPECT_EQ(std::memcmp(output.data(), expected.data(), expected.size()),
                  0);
    }

    void WaitForReadLeaseExpiry() {
        // The verification read holds a lease too. Allow the master's deadline
        // to pass before testing the successful synchronous write path.
        std::this_thread::sleep_until(last_read_lease_deadline_ +
                                      std::chrono::milliseconds(100));
    }

    static std::vector<std::vector<Slice>> MakeSlices(
        std::vector<std::string>& values) {
        std::vector<std::vector<Slice>> slices;
        slices.reserve(values.size());
        for (auto& value : values) {
            slices.push_back({Slice{value.data(), value.size()}});
        }
        return slices;
    }

    void SetEnv(const std::string& key, const std::string& value) {
        const char* old_value = ::getenv(key.c_str());
        saved_env_.push_back({key, old_value
                                       ? std::optional<std::string>(old_value)
                                       : std::nullopt});
        ::setenv(key.c_str(), value.c_str(), 1);
    }

    void RestoreEnv() {
        for (auto it = saved_env_.rbegin(); it != saved_env_.rend(); ++it) {
            if (it->second) {
                ::setenv(it->first.c_str(), it->second->c_str(), 1);
            } else {
                ::unsetenv(it->first.c_str());
            }
        }
        saved_env_.clear();
    }

    inline static std::atomic<int> next_root_{0};
    testing::InProcMaster master_;
    std::shared_ptr<Client> writer_;
    std::shared_ptr<Client> provider_;
    std::shared_ptr<DistributedStorageBackend> backend_;
    FailingPosixFsAdapter* adapter_ = nullptr;
    void* segment_ = nullptr;
    size_t segment_size_ = 0;
    std::string root_;
    std::chrono::steady_clock::time_point last_read_lease_deadline_{};
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_env_;
};

class DfsParallelClientTest : public DfsSyncClientTest,
                              public ::testing::WithParamInterface<bool> {
   protected:
    virtual bool OverlapEnabled() const { return true; }
    void SetUp() override {
        SetEnv("MC_STORE_DFS_WRITE_FENCING", "1");
        SetEnv("MC_STORE_PARALLEL_DFS_PUT", OverlapEnabled() ? "1" : "0");
        SetEnv("MC_STORE_MEMCPY", "1");
        DfsSyncClientTest::SetUp();
        ASSERT_TRUE(provider_->UnmountSegment(segment_, segment_size_));
        ASSERT_TRUE(writer_->MountSegment(segment_, segment_size_, "tcp"));
    }
    void TearDown() override {
        blocking_device.Release();
        blocking_device.Disarm();
        if (writer_) (void)writer_->UnmountSegment(segment_, segment_size_);
        DfsSyncClientTest::TearDown();
    }
    void RunWrite(bool fail_memory, bool fail_dfs, bool fail_staging = false) {
        if (ClientObjectChecksumConfig::IsEnabledAtFirstUse()) {
            GTEST_SKIP()
                << "The ordering test requires checksum staging disabled";
        }
        std::vector<std::string> keys{"parallel_order"};
        std::vector<std::string> values{std::string(4096, 'O')};
        auto slices = MakeSlices(values);
        auto config = DfsConfig();
        config.prefer_alloc_in_same_node = GetParam();
        blocking_device.Arm(values[0].data(), segment_, segment_size_,
                            fail_memory, fail_staging);
        if (fail_dfs) adapter_->FailWriteCall(adapter_->WriteCalls() + 1);
        auto result = std::async(std::launch::async, [&] {
            return writer_->BatchPut(keys, slices, config);
        });
        const bool memory_started = blocking_device.WaitForMemory();
        const bool staging_overlapped = blocking_device.WaitForStaging();
        EXPECT_TRUE(memory_started);
        EXPECT_EQ(staging_overlapped, OverlapEnabled());
        EXPECT_EQ(result.wait_for(std::chrono::milliseconds(0)),
                  std::future_status::timeout);
        // A second start must still see the original processing write even
        // when DFS already failed: no revoke while MEMORY reads the source.
        auto competing = writer_->BatchPut(keys, slices, config);
        EXPECT_EQ(competing.size(), 1);
        if (competing.size() == 1) {
            EXPECT_FALSE(competing[0]);
            if (!competing[0])
                EXPECT_EQ(competing[0].error(), ErrorCode::OBJECT_REPLICA_BUSY);
        }
        blocking_device.Release();
        auto results = result.get();
        blocking_device.Disarm();
        ASSERT_EQ(results.size(), 1);
        if (fail_memory || fail_dfs || fail_staging) {
            ASSERT_FALSE(results[0]);
            EXPECT_EQ(results[0].error(), fail_memory || fail_staging
                                              ? ErrorCode::TRANSFER_FAIL
                                              : ErrorCode::FILE_WRITE_FAIL);
            EXPECT_FALSE(writer_->Query(keys[0]));
        } else {
            ASSERT_TRUE(results[0]);
            ExpectDfsValue(keys[0], values[0]);
            auto query = writer_->Query(keys[0]);
            ASSERT_TRUE(query);
            EXPECT_EQ(query->replicas.size(), 2);
            std::string output(values[0].size(), '\0');
            std::vector<Slice> output_slices{{output.data(), output.size()}};
            ASSERT_TRUE(writer_->Get(keys[0], *query, output_slices));
            EXPECT_EQ(output, values[0]);
        }
    }
};

TEST_P(DfsParallelClientTest, BothSucceed) { RunWrite(false, false); }
TEST_P(DfsParallelClientTest, MemoryFailureRevokesSuccessfulDfs) {
    RunWrite(true, false);
}
TEST_P(DfsParallelClientTest, DfsFailureDrainsMemoryBeforeRevoke) {
    RunWrite(false, true);
}
TEST_P(DfsParallelClientTest, BothFailPreserveMemoryError) {
    RunWrite(true, true);
}
TEST_P(DfsParallelClientTest, StagingFailureDrainsMemory) {
    RunWrite(false, false, true);
}
INSTANTIATE_TEST_SUITE_P(PlainAndMerged, DfsParallelClientTest,
                         ::testing::Values(false, true));

class DfsSerialClientTest : public DfsParallelClientTest {
   protected:
    bool OverlapEnabled() const override { return false; }
};
TEST_P(DfsSerialClientTest, DisabledModeWaitsForMemoryBeforeStaging) {
    RunWrite(false, false);
}
INSTANTIATE_TEST_SUITE_P(PlainAndMerged, DfsSerialClientTest,
                         ::testing::Values(false, true));

class DfsLifetimeClientTest : public DfsSyncClientTest,
                              public ::testing::WithParamInterface<bool> {
   protected:
    void SetUp() override {
        SetEnv("MC_STORE_DFS_WRITE_FENCING", "1");
        SetEnv("MC_STORE_PARALLEL_DFS_PUT", GetParam() ? "1" : "0");
        DfsSyncClientTest::SetUp();
    }
};

// Run explicitly as the distributed lifetime acceptance gate. It is separate
// from the scheduling regressions because upstream currently preempts DFS
// writes.
TEST_P(DfsLifetimeClientTest, LateWriterMustNotOverwriteUpsertedDfsValue) {
    struct ReleaseOnExit {
        FailingPosixFsAdapter* adapter;
        ~ReleaseOnExit() { adapter->ReleaseWrite(); }
    };
    SetEnv("MC_STORE_PARALLEL_DFS_PUT", GetParam() ? "1" : "0");
    writer_ = CreateClient("127.0.0.1:18103");
    ASSERT_NE(writer_, nullptr);
    writer_->SetDfsStorageBackend(backend_);

    FileStorageConfig file_config;
    file_config.storage_backend_type = StorageBackendType::kDistributed;
    file_config.storage_filepath = root_;
    DistributedStorageConfig config;
    config.fsdir = root_;
    config.fs_adapter_type = "posix";
    config.shard_count = 2;
    config.shard_capacity = 16 * 1024 * 1024;
    config.alignment = 4096;
    auto replacement_backend = std::make_shared<DistributedStorageBackend>(
        file_config, config, std::make_unique<PosixFsAdapter>());
    ASSERT_TRUE(replacement_backend->Init());
    provider_->SetDfsStorageBackend(replacement_backend);

    const std::string key = "late_writer_gate";
    std::string old_value(4096, 'A'), new_value(4096, 'B');
    std::vector<std::vector<Slice>> old_slices{
        {{old_value.data(), old_value.size()}}};
    adapter_->BlockWriteCall(adapter_->WriteCalls() + 1);
    auto old_result = std::async(std::launch::async, [&] {
        return writer_->BatchPut({key}, old_slices, DfsConfig());
    });
    ReleaseOnExit release{adapter_};
    const bool blocked = adapter_->WaitForBlockedWrite();
    if (!blocked) adapter_->ReleaseWrite();
    ASSERT_TRUE(blocked);
    std::vector<Slice> new_slices{{new_value.data(), new_value.size()}};
    const auto replacement = provider_->Upsert(key, new_slices, DfsConfig());
    EXPECT_FALSE(replacement);
    if (!replacement)
        EXPECT_EQ(replacement.error(), ErrorCode::OBJECT_REPLICA_BUSY);
    const auto old_offset = adapter_->BlockedOffset();
    adapter_->ReleaseWrite();
    const auto previous = old_result.get();
    ASSERT_EQ(previous.size(), 1);
    ASSERT_TRUE(previous[0]);
    ASSERT_TRUE(provider_->Upsert(key, new_slices, DfsConfig()));
    auto query = provider_->Query(key);
    ASSERT_TRUE(query);
    auto dfs = std::find_if(
        query->replicas.begin(), query->replicas.end(),
        [](const auto& replica) { return replica.is_dfs_replica(); });
    ASSERT_NE(dfs, query->replicas.end());
    const auto descriptor = dfs->get_dfs_descriptor();

    std::string output(new_value.size(), '\0');
    auto reads = replacement_backend->BatchRead(
        {{key, descriptor, {{output.data(), output.size()}}}});
    ASSERT_EQ(reads.size(), 1);
    ASSERT_TRUE(reads[0]);
    EXPECT_EQ(output.front(), 'B');
    EXPECT_TRUE(output == new_value)
        << "old offset=" << old_offset
        << ", replacement offset=" << descriptor.offset;
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, DfsLifetimeClientTest,
                         ::testing::Values(false, true));

class DfsFencedProtocolTest : public DfsSyncClientTest {
   protected:
    void SetUp() override {
        SetEnv("MC_STORE_DFS_WRITE_FENCING", "1");
        DfsSyncClientTest::SetUp();
    }
};

TEST_F(DfsFencedProtocolTest, StaleStartEndAndRevokeCannotTouchANewerAttempt) {
    const UUID owner = generate_uuid();
    MasterClient client(owner);
    ASSERT_EQ(client.Connect(master_.master_address()), ErrorCode::OK);
    const std::vector<std::string> keys{"attempt_replay"};
    const std::vector<std::vector<uint64_t>> sizes{{4096}};
    const UUID first{1, owner.second}, second{2, owner.second},
        third{3, owner.second};
    auto starts =
        client.BatchStartFencedDfsWrite(keys, sizes, DfsConfig(), first, false);
    ASSERT_EQ(starts.size(), 1);
    ASSERT_TRUE(starts[0]);
    auto duplicate =
        client.BatchStartFencedDfsWrite(keys, sizes, DfsConfig(), first, false);
    ASSERT_EQ(duplicate.size(), 1);
    ASSERT_TRUE(duplicate[0]);
    EXPECT_EQ(duplicate[0]->size(), starts[0]->size());
    std::vector<ObjectMeta> objects{{keys[0], std::nullopt}};
    auto wrong_end = client.BatchEndFencedDfsWrite(objects, second);
    ASSERT_EQ(wrong_end.size(), 1);
    ASSERT_FALSE(wrong_end[0]);
    EXPECT_EQ(wrong_end[0].error(), ErrorCode::INVALID_WRITE);
    auto legacy_end = client.BatchPutEnd(objects, ReplicaType::ALL);
    ASSERT_EQ(legacy_end.size(), 1);
    ASSERT_FALSE(legacy_end[0]);
    EXPECT_EQ(legacy_end[0].error(), ErrorCode::INVALID_WRITE);
    // No data I/O is submitted in this control-plane test; this Revoke is
    // the acknowledgement that the descriptor has no outstanding users.
    ASSERT_TRUE(client.BatchRevokeFencedDfsWrite(keys, first)[0]);
    ASSERT_TRUE(client.BatchStartFencedDfsWrite(keys, sizes, DfsConfig(),
                                                second, false)[0]);
    auto late_end = client.BatchEndFencedDfsWrite(objects, first);
    auto late_revoke = client.BatchRevokeFencedDfsWrite(keys, first);
    ASSERT_FALSE(late_end[0]);
    ASSERT_FALSE(late_revoke[0]);
    EXPECT_EQ(late_end[0].error(), ErrorCode::INVALID_WRITE);
    EXPECT_EQ(late_revoke[0].error(), ErrorCode::INVALID_WRITE);
    ASSERT_TRUE(client.BatchRevokeFencedDfsWrite(keys, second)[0]);
    auto retired_start =
        client.BatchStartFencedDfsWrite(keys, sizes, DfsConfig(), first, true);
    ASSERT_FALSE(retired_start[0]);
    EXPECT_EQ(retired_start[0].error(), ErrorCode::INVALID_WRITE);
    ASSERT_TRUE(client.BatchStartFencedDfsWrite(keys, sizes, DfsConfig(), third,
                                                false)[0]);
    ASSERT_TRUE(client.BatchRevokeFencedDfsWrite(keys, third)[0]);
}

TEST_F(DfsFencedProtocolTest,
       LegacyDfsWritesAreRejectedAndMemoryOnlyStillWorks) {
    const UUID owner = generate_uuid();
    MasterClient client(owner);
    ASSERT_EQ(client.Connect(master_.master_address()), ErrorCode::OK);
    auto legacy = client.BatchPutStart({"legacy_dfs"}, {{4096}}, DfsConfig());
    ASSERT_EQ(legacy.size(), 1);
    ASSERT_FALSE(legacy[0]);
    EXPECT_EQ(legacy[0].error(), ErrorCode::NOT_SUPPORTED);
    ReplicateConfig memory;
    std::string value(4096, 'M');
    std::vector<Slice> slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put("memory_only", slices, memory));
    auto query = writer_->Query("memory_only");
    ASSERT_TRUE(query);
    std::string output(value.size(), '\0');
    std::vector<Slice> reads{{output.data(), output.size()}};
    ASSERT_TRUE(writer_->Get("memory_only", *query, reads));
    EXPECT_EQ(output, value);
}

TEST_F(DfsFencedProtocolTest,
       ReplacingBackendDoesNotDestroyAnInflightWriteOwner) {
    const std::vector<std::string> keys{"backend_owner"};
    std::vector<std::string> values{std::string(4096, 'L')};
    auto slices = MakeSlices(values);
    adapter_->BlockWriteCall(adapter_->WriteCalls() + 1);
    auto result = std::async(std::launch::async, [&] {
        return writer_->BatchPut(keys, slices, DfsConfig());
    });
    struct ReleaseOnExit {
        FailingPosixFsAdapter* adapter;
        ~ReleaseOnExit() {
            if (adapter) adapter->ReleaseWrite();
        }
    } release{adapter_};
    ASSERT_TRUE(adapter_->WaitForBlockedWrite());
    std::weak_ptr<DistributedStorageBackend> owner = backend_;
    writer_->SetDfsStorageBackend(nullptr);
    backend_.reset();
    EXPECT_FALSE(owner.expired());
    adapter_->ReleaseWrite();
    release.adapter = nullptr;
    ASSERT_TRUE(result.get()[0]);
    EXPECT_TRUE(owner.expired());
    adapter_ = nullptr;
}

TEST_F(DfsFencedProtocolTest,
       HistoryBudgetFailsClosedWithoutBlockingMemoryOnly) {
    const UUID owner = generate_uuid();
    const std::string key = "history_budget";
    MasterServiceTestPeer peer(*master_.service());
    peer.FillFencedWriteHistoryForKey(key, owner);
    MasterClient client(owner);
    ASSERT_EQ(client.Connect(master_.master_address()), ErrorCode::OK);
    auto result = client.BatchStartFencedDfsWrite({key}, {{4096}}, DfsConfig(),
                                                  {1, owner.second}, false);
    ASSERT_FALSE(result[0]);
    EXPECT_EQ(result[0].error(), ErrorCode::NO_AVAILABLE_HANDLE);
    ReplicateConfig memory;
    auto legacy = client.BatchPutStart({key}, {{4096}}, memory);
    ASSERT_TRUE(legacy[0]);
    ASSERT_TRUE(client.BatchPutRevoke({key}, ReplicaType::ALL)[0]);
}

class DfsFencedTimeoutTest : public DfsFencedProtocolTest {};
TEST_F(DfsFencedTimeoutTest,
       ExpirationDoesNotRevokeOrReuseAnUnaccountedWriter) {
    const UUID owner = generate_uuid();
    MasterClient client(owner);
    ASSERT_EQ(client.Connect(master_.master_address()), ErrorCode::OK);
    const UUID first{1, owner.second}, second{2, owner.second};
    auto first_start = client.BatchStartFencedDfsWrite(
        {"unaccounted"}, {{4096}}, DfsConfig(), first, false);
    ASSERT_TRUE(first_start[0]);
    auto descriptor = [](const auto& replicas) {
        auto dfs = std::find_if(
            replicas.begin(), replicas.end(),
            [](const auto& replica) { return replica.is_dfs_replica(); });
        return dfs->get_dfs_descriptor();
    };
    const auto original = descriptor(*first_start[0]);
    // Drive the actual cleanup with a future timestamp under its shard locks.
    MasterServiceTestPeer peer(*master_.service());
    peer.ExpireProcessingAt(std::chrono::system_clock::now() +
                            std::chrono::hours(1));
    auto retry = client.BatchStartFencedDfsWrite({"unaccounted"}, {{4096}},
                                                 DfsConfig(), second, false);
    ASSERT_FALSE(retry[0]);
    EXPECT_EQ(retry[0].error(), ErrorCode::OBJECT_REPLICA_BUSY);
    auto other = client.BatchStartFencedDfsWrite({"other"}, {{4096}},
                                                 DfsConfig(), second, false);
    ASSERT_TRUE(other[0]);
    const auto allocated = descriptor(*other[0]);
    EXPECT_TRUE(original.shard_idx != allocated.shard_idx ||
                original.offset != allocated.offset);
    // This test submits no data I/O, so releasing these descriptors is safe.
    ASSERT_TRUE(client.BatchRevokeFencedDfsWrite({"unaccounted"}, first)[0]);
    ASSERT_TRUE(client.BatchRevokeFencedDfsWrite({"other"}, second)[0]);
}

TEST_F(DfsFencedProtocolTest, NamespaceSealRejectsRestartAndLegacyReuse) {
    writer_.reset();
    provider_.reset();
    master_.Stop();
    EXPECT_THROW({ MasterService restarted; }, std::invalid_argument);
    SetEnv("MC_STORE_DFS_WRITE_FENCING", "0");
    EXPECT_THROW({ MasterService legacy; }, std::invalid_argument);
}

#if defined(USE_3FS) && defined(USE_CUDA)
class DfsNativeCudaClientTest : public DfsSyncClientTest {
   protected:
    std::filesystem::path TestRoot() const override {
        return std::getenv("MOONCAKE_NATIVE_TEST_ROOT");
    }
    const char* AdapterName() const override { return "hf3fs"; }
    std::unique_ptr<FileSystemAdapter> CreateAdapter() override {
        return std::make_unique<Hf3fsAdapter>();
    }
    void SetUp() override {
        if (!std::getenv("MOONCAKE_NATIVE_TEST_ROOT")) {
            GTEST_SKIP()
                << "Set MOONCAKE_NATIVE_TEST_ROOT to a writable 3FS mount";
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        SetEnv("MC_STORE_DFS_WRITE_FENCING", "1");
        SetEnv("MC_STORE_PARALLEL_DFS_PUT", "1");
        SetEnv("MC_STORE_MEMCPY", "1");
        DfsSyncClientTest::SetUp();
        ASSERT_TRUE(provider_->UnmountSegment(segment_, segment_size_));
        ASSERT_TRUE(writer_->MountSegment(segment_, segment_size_, "tcp"));
    }
    void TearDown() override {
        if (writer_) (void)writer_->UnmountSegment(segment_, segment_size_);
        DfsSyncClientTest::TearDown();
    }
};
TEST_F(DfsNativeCudaClientTest, GpuSourceReachesMemoryAndNativeDfs) {
    const std::string key = "native_cuda_parallel_put";
    std::string value(1024 * 1024, 'G');
    void* source = nullptr;
    ASSERT_EQ(cudaMalloc(&source, value.size()), cudaSuccess);
    struct FreeOnExit {
        void* source;
        ~FreeOnExit() { cudaFree(source); }
    } owner{source};
    ASSERT_EQ(
        cudaMemcpy(source, value.data(), value.size(), cudaMemcpyHostToDevice),
        cudaSuccess);
    std::vector<std::vector<Slice>> slices{{{source, value.size()}}};
    const auto results = writer_->BatchPut({key}, slices, DfsConfig());
    ASSERT_EQ(results.size(), 1);
    ASSERT_TRUE(results[0]);
    ExpectDfsValue(key, value);
    auto query = writer_->Query(key);
    ASSERT_TRUE(query);
    EXPECT_EQ(query->replicas.size(), 2);
    std::string memory_value(value.size(), '\0');
    std::vector<Slice> output{{memory_value.data(), memory_value.size()}};
    ASSERT_TRUE(writer_->Get(key, *query, output));
    EXPECT_EQ(memory_value, value);
}
#endif

TEST_F(DfsSyncClientTest, PutAndUpsertReturnAfterDfsWrite) {
    const std::string key = "sync_put_upsert";
    std::string initial(4096, 'A');
    std::vector<Slice> initial_slices{{initial.data(), initial.size()}};
    ASSERT_TRUE(writer_->Put(key, initial_slices, DfsConfig()).has_value());
    ExpectDfsValue(key, initial);

    std::string updated(4096, 'B');
    std::vector<Slice> updated_slices{{updated.data(), updated.size()}};
    auto leased_upsert = writer_->Upsert(key, updated_slices, DfsConfig());
    ASSERT_FALSE(leased_upsert.has_value());
    EXPECT_EQ(leased_upsert.error(), ErrorCode::OBJECT_HAS_LEASE);
    WaitForReadLeaseExpiry();
    ASSERT_TRUE(writer_->Upsert(key, updated_slices, DfsConfig()).has_value());
    ExpectDfsValue(key, updated);
}

TEST_F(DfsSyncClientTest, BatchPutAndBatchUpsertReturnAfterDfsWrite) {
    std::vector<std::string> keys{"sync_batch_0", "sync_batch_1"};
    std::vector<std::string> initial{std::string(4096, 'C'),
                                     std::string(4096, 'D')};
    auto initial_slices = MakeSlices(initial);
    auto put_results = writer_->BatchPut(keys, initial_slices, DfsConfig());
    ASSERT_EQ(put_results.size(), keys.size());
    ASSERT_TRUE(put_results[0].has_value());
    ASSERT_TRUE(put_results[1].has_value());
    ExpectDfsValue(keys[0], initial[0]);
    ExpectDfsValue(keys[1], initial[1]);

    std::vector<std::string> updated{std::string(4096, 'E'),
                                     std::string(4096, 'F')};
    auto updated_slices = MakeSlices(updated);
    auto leased_upserts =
        writer_->BatchUpsert(keys, updated_slices, DfsConfig());
    ASSERT_EQ(leased_upserts.size(), keys.size());
    for (const auto& result : leased_upserts) {
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::OBJECT_HAS_LEASE);
    }
    WaitForReadLeaseExpiry();
    auto upsert_results =
        writer_->BatchUpsert(keys, updated_slices, DfsConfig());
    ASSERT_EQ(upsert_results.size(), keys.size());
    ASSERT_TRUE(upsert_results[0].has_value());
    ASSERT_TRUE(upsert_results[1].has_value());
    ExpectDfsValue(keys[0], updated[0]);
    ExpectDfsValue(keys[1], updated[1]);
}

TEST_F(DfsSyncClientTest, PreferSameNodeBatchPutReturnsAfterDfsWrite) {
    std::vector<std::string> keys{"sync_same_node_0", "sync_same_node_1"};
    std::vector<std::string> values{std::string(4096, 'J'),
                                    std::string(4096, 'K')};
    auto slices = MakeSlices(values);
    auto config = DfsConfig();
    config.prefer_alloc_in_same_node = true;

    auto results = writer_->BatchPut(keys, slices, config);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_TRUE(results[0].has_value());
    ASSERT_TRUE(results[1].has_value());
    ExpectDfsValue(keys[0], values[0]);
    ExpectDfsValue(keys[1], values[1]);
}

TEST_F(DfsSyncClientTest, BatchDfsFailureOnlyRevokesFailedKey) {
    std::vector<std::string> keys{"sync_success", "sync_failure"};
    std::vector<std::string> values{std::string(4096, 'G'),
                                    std::string(4096, 'H')};
    auto slices = MakeSlices(values);
    adapter_->FailWriteCall(adapter_->WriteCalls() + 2);

    auto results = writer_->BatchPut(keys, slices, DfsConfig());
    ASSERT_EQ(results.size(), keys.size());
    EXPECT_TRUE(results[0].has_value());
    ASSERT_FALSE(results[1].has_value());
    EXPECT_EQ(results[1].error(), ErrorCode::FILE_WRITE_FAIL);
    ExpectDfsValue(keys[0], values[0]);

    auto failed_query = writer_->Query(keys[1]);
    ASSERT_FALSE(failed_query.has_value());
    EXPECT_EQ(failed_query.error(), ErrorCode::OBJECT_NOT_FOUND);
}

TEST_F(DfsSyncClientTest, MissingDfsBackendRevokesObject) {
    auto client_without_backend = CreateClient("127.0.0.1:18103");
    ASSERT_NE(client_without_backend, nullptr);
    std::string value(4096, 'I');
    std::vector<Slice> slices{{value.data(), value.size()}};

    auto result =
        client_without_backend->Put("sync_no_backend", slices, DfsConfig());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
    auto query = client_without_backend->Query("sync_no_backend");
    ASSERT_FALSE(query.has_value());
    EXPECT_EQ(query.error(), ErrorCode::OBJECT_NOT_FOUND);
}

TEST_F(DfsSyncClientTest, ExistingClientUsesOnlineExpandedDfsCapacity) {
    const std::string old_key = "before_expansion";
    std::string old_value(4096, 'O');
    std::vector<Slice> old_slices{{old_value.data(), old_value.size()}};
    ASSERT_TRUE(writer_->Put(old_key, old_slices, DfsConfig()));
    auto old_query = QueryDfsOnly(old_key);
    ASSERT_TRUE(old_query);
    const auto old_path = old_query->replicas[0].get_dfs_descriptor().file_path;
    auto expanded = master_.service()->ExpandDfsShards(4);
    ASSERT_TRUE(expanded);
    EXPECT_EQ(*expanded, 4);

    std::string new_key;
    for (int i = 0; i < 1000; ++i) {
        new_key = "after_expansion_" + std::to_string(i);
        if (std::hash<std::string>{}(new_key) % 4 == 3) break;
    }
    ASSERT_EQ(std::hash<std::string>{}(new_key) % 4, 3);
    std::string value(4096, 'N');
    std::vector<Slice> write_slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put(new_key, write_slices, DfsConfig()));
    auto query = QueryDfsOnly(new_key);
    ASSERT_TRUE(query);
    EXPECT_EQ(query->replicas[0].get_dfs_descriptor().shard_idx, 3);
    std::string output(value.size(), '\0');
    std::vector<Slice> read_slices{{output.data(), output.size()}};
    ASSERT_TRUE(writer_->Get(new_key, *query, read_slices));
    EXPECT_EQ(output, value);
    ExpectDfsValue(old_key, old_value);
    auto after = QueryDfsOnly(old_key);
    ASSERT_TRUE(after);
    EXPECT_EQ(after->replicas[0].get_dfs_descriptor().file_path, old_path);
}

TEST_F(DfsSyncClientTest, GetReadsDfsIntoMultipleSlices) {
    const std::string key = "dfs_multi_slice_get";
    std::string value(4096, 'M');
    std::vector<Slice> write_slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put(key, write_slices, DfsConfig()).has_value());

    auto query = QueryDfsOnly(key);
    ASSERT_TRUE(query.has_value());
    std::vector<char> first(1024), second(3072);
    std::vector<Slice> read_slices{{first.data(), first.size()},
                                   {second.data(), second.size()}};
    ASSERT_TRUE(writer_->Get(key, *query, read_slices).has_value());
    EXPECT_EQ(std::memcmp(value.data(), first.data(), first.size()), 0);
    EXPECT_EQ(
        std::memcmp(value.data() + first.size(), second.data(), second.size()),
        0);
}

TEST_F(DfsSyncClientTest, BatchGetUsesExplicitDfsDescriptors) {
    std::vector<std::string> keys{"dfs_batch_read_0", "dfs_batch_read_1"};
    std::vector<std::string> values{std::string(4096, 'N'),
                                    std::string(4096, 'O')};
    auto write_slices = MakeSlices(values);
    auto put_results = writer_->BatchPut(keys, write_slices, DfsConfig());
    ASSERT_EQ(put_results.size(), keys.size());
    ASSERT_TRUE(put_results[0].has_value());
    ASSERT_TRUE(put_results[1].has_value());

    std::vector<QueryResult> queries;
    queries.reserve(keys.size());
    for (const auto& key : keys) {
        auto query = QueryDfsOnly(key);
        ASSERT_TRUE(query.has_value());
        queries.push_back(*query);
    }

    std::vector<std::string> output{std::string(4096, '\0'),
                                    std::string(4096, '\0')};
    std::unordered_map<std::string, std::vector<Slice>> read_slices;
    for (size_t i = 0; i < keys.size(); ++i) {
        read_slices[keys[i]] = {{output[i].data(), output[i].size()}};
    }
    auto get_results = writer_->BatchGet(keys, queries, read_slices);
    ASSERT_EQ(get_results.size(), keys.size());
    ASSERT_TRUE(get_results[0].has_value());
    ASSERT_TRUE(get_results[1].has_value());
    EXPECT_EQ(output[0], values[0]);
    EXPECT_EQ(output[1], values[1]);
}

TEST_F(DfsSyncClientTest, BatchGetVerifiesDfsChecksum) {
    if (!ClientObjectChecksumConfig::IsEnabledAtFirstUse()) {
        GTEST_SKIP() << "MOONCAKE_STORE_CHECKSUM is not enabled";
    }

    const std::string key = "dfs_batch_checksum";
    std::string value(4096, 'P');
    std::vector<Slice> write_slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put(key, write_slices, DfsConfig()).has_value());

    auto query = QueryDfsOnly(key);
    ASSERT_TRUE(query.has_value());
    ASSERT_TRUE(query->object_checksum.has_value());
    std::vector<Replica::Descriptor> replicas = query->replicas;
    std::vector<QueryResult> queries;
    queries.emplace_back(
        std::move(replicas), query->lease_timeout,
        std::optional<uint64_t>(*query->object_checksum ^ uint64_t{1}));

    std::string output(value.size(), '\0');
    std::unordered_map<std::string, std::vector<Slice>> read_slices;
    read_slices[key] = {{output.data(), output.size()}};
    auto results = writer_->BatchGet({key}, queries, read_slices);
    ASSERT_EQ(results.size(), 1);
    ASSERT_FALSE(results[0].has_value());
    EXPECT_EQ(results[0].error(), ErrorCode::CHECKSUM_MISMATCH);
}

TEST_F(DfsSyncClientTest, GetVerifiesDfsChecksumAndRecordsError) {
    if (!ClientObjectChecksumConfig::IsEnabledAtFirstUse()) {
        GTEST_SKIP() << "MOONCAKE_STORE_CHECKSUM is not enabled";
    }
    auto* metric = writer_->GetDfsMetricPtr();
    if (metric == nullptr) {
        GTEST_SKIP() << "Client metrics are disabled";
    }

    const std::string key = "dfs_single_checksum";
    std::string value(4096, 'V');
    std::vector<Slice> write_slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put(key, write_slices, DfsConfig()).has_value());

    auto query = QueryDfsOnly(key);
    ASSERT_TRUE(query.has_value());
    ASSERT_TRUE(query->object_checksum.has_value());

    const std::array<std::string, 1> mismatch_label{
        toString(ErrorCode::CHECKSUM_MISMATCH)};
    const int64_t base_read_ops = metric->dfs_read_ops.value();
    const int64_t base_errors = metric->dfs_read_errors.value(mismatch_label);

    const std::optional<uint64_t> bad_checksum(*query->object_checksum ^
                                               uint64_t{1});
    std::vector<Replica::Descriptor> replicas = query->replicas;
    QueryResult corrupted(std::move(replicas), query->lease_timeout,
                          bad_checksum);

    std::vector<char> output(value.size());
    std::vector<Slice> read_slices{{output.data(), output.size()}};
    auto result = writer_->Get(key, corrupted, read_slices);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHECKSUM_MISMATCH);

    // The I/O completed, so it still counts as a read op, exactly as in the
    // batch path; the rejection lands in the error family instead.
    EXPECT_EQ(metric->dfs_read_ops.value(), base_read_ops + 1);
    EXPECT_EQ(metric->dfs_read_errors.value(mismatch_label), base_errors + 1);
}

TEST_F(DfsSyncClientTest, DfsMetricsCountSuccessfulWriteAndRead) {
    auto* metric = writer_->GetDfsMetricPtr();
    if (metric == nullptr) {
        GTEST_SKIP() << "Client metrics are disabled";
    }
    const int64_t base_write_ops = metric->dfs_write_ops.value();
    const int64_t base_write_bytes = metric->dfs_write_bytes.value();
    const int64_t base_read_ops = metric->dfs_read_ops.value();
    const int64_t base_read_bytes = metric->dfs_read_bytes.value();

    const std::string key = "dfs_metric_success";
    std::string value(4096, 'Q');
    std::vector<Slice> write_slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put(key, write_slices, DfsConfig()).has_value());

    EXPECT_EQ(metric->dfs_write_ops.value(), base_write_ops + 1);
    EXPECT_EQ(metric->dfs_write_bytes.value(),
              base_write_bytes + static_cast<int64_t>(value.size()));

    auto query = QueryDfsOnly(key);
    ASSERT_TRUE(query.has_value());
    std::vector<char> output(value.size());
    std::vector<Slice> read_slices{{output.data(), output.size()}};
    ASSERT_TRUE(writer_->Get(key, *query, read_slices).has_value());

    EXPECT_EQ(metric->dfs_read_ops.value(), base_read_ops + 1);
    EXPECT_EQ(metric->dfs_read_bytes.value(),
              base_read_bytes + static_cast<int64_t>(value.size()));
    // Host-resident slices never go through the D2H staging path.
    EXPECT_EQ(metric->dfs_writes_skipped.value(), 0);
}

TEST_F(DfsSyncClientTest, DfsMetricsCountWriteFailureSeparatelyFromOps) {
    auto* metric = writer_->GetDfsMetricPtr();
    if (metric == nullptr) {
        GTEST_SKIP() << "Client metrics are disabled";
    }
    const std::array<std::string, 1> write_fail_label{
        toString(ErrorCode::FILE_WRITE_FAIL)};
    const int64_t base_write_ops = metric->dfs_write_ops.value();
    const int64_t base_errors =
        metric->dfs_write_errors.value(write_fail_label);

    std::vector<std::string> keys{"dfs_metric_ok", "dfs_metric_fail"};
    std::vector<std::string> values{std::string(4096, 'R'),
                                    std::string(4096, 'S')};
    auto slices = MakeSlices(values);
    adapter_->FailWriteCall(adapter_->WriteCalls() + 2);

    auto results = writer_->BatchPut(keys, slices, DfsConfig());
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_TRUE(results[0].has_value());
    ASSERT_FALSE(results[1].has_value());

    // The failed key must not inflate ops/bytes; it lands in the error family
    // instead, keyed by the ErrorCode name.
    EXPECT_EQ(metric->dfs_write_ops.value(), base_write_ops + 1);
    EXPECT_EQ(metric->dfs_write_errors.value(write_fail_label),
              base_errors + 1);
}

TEST_F(DfsSyncClientTest, DfsMetricsRecordUnavailableBackend) {
    auto client_without_backend = CreateClient("127.0.0.1:18104");
    ASSERT_NE(client_without_backend, nullptr);
    auto* metric = client_without_backend->GetDfsMetricPtr();
    if (metric == nullptr) {
        GTEST_SKIP() << "Client metrics are disabled";
    }
    const std::array<std::string, 1> unavailable_label{
        toString(ErrorCode::DFS_SERVICE_UNAVAILABLE)};
    ASSERT_EQ(metric->dfs_write_errors.value(unavailable_label), 0);

    std::string value(4096, 'T');
    std::vector<Slice> slices{{value.data(), value.size()}};
    auto result = client_without_backend->Put("dfs_metric_no_backend", slices,
                                              DfsConfig());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
    EXPECT_EQ(metric->dfs_write_errors.value(unavailable_label), 1);
    EXPECT_EQ(metric->dfs_write_ops.value(), 0);
}

TEST_F(DfsSyncClientTest, DfsMetricsAppearInSerializedOutput) {
    auto* metric = writer_->GetDfsMetricPtr();
    if (metric == nullptr) {
        GTEST_SKIP() << "Client metrics are disabled";
    }
    const std::string key = "dfs_metric_serialize";
    std::string value(4096, 'U');
    std::vector<Slice> write_slices{{value.data(), value.size()}};
    ASSERT_TRUE(writer_->Put(key, write_slices, DfsConfig()).has_value());

    auto serialized = writer_->SerializeMetrics();
    ASSERT_TRUE(serialized.has_value());
    EXPECT_NE(serialized->find("mooncake_dfs_write_bytes_total"),
              std::string::npos);
    EXPECT_NE(serialized->find("mooncake_dfs_write_ops_total"),
              std::string::npos);

    auto summary = writer_->GetSummaryMetrics();
    ASSERT_TRUE(summary.has_value());
    EXPECT_NE(summary->find("=== DFS Metrics Summary ==="), std::string::npos);
}

}  // namespace mooncake::test
