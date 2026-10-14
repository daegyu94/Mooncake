#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <future>
#include <chrono>
#include <vector>
#include <utility>

#include "hf3fs/hf3fs.h"
#include "storage/distributed/hf3fs_adapter.h"

// Only the native C API is replaced. The production adapter and resource
// manager must retain their buffers until every prepared write is reaped.
namespace {
struct PendingWrite {
    int index;
    void* buffer;
    size_t bytes;
    const void* userdata;
    bool read;
};
struct Ring {
    std::deque<PendingWrite> writes;
    int next_index = 0;
};
struct Faults {
    int submit_error = 0;
    std::deque<int> waits;
    bool unexpected_completion = false;
    int64_t result = -1;
    bool fail_prepare = false;
    size_t completed = 0;
    std::vector<char> data;
} faults;
std::mutex wait_mutex;
std::condition_variable wait_cv;
bool block_wait = false;
bool wait_entered = false;
bool wait_released = false;
}  // namespace

extern "C" {
int hf3fs_extract_mount_point(char*, int, const char*) { return -1; }
int hf3fs_reg_fd(int, uint64_t) { return 0; }
void hf3fs_dereg_fd(int) {}
int hf3fs_iovcreate(hf3fs_iov* iov, const char*, size_t size, size_t, int) {
    *iov = {};
    iov->base = static_cast<uint8_t*>(std::malloc(size));
    iov->size = size;
    return iov->base ? 0 : -ENOMEM;
}
void hf3fs_iovdestroy(hf3fs_iov* iov) { std::free(iov->base); }
int hf3fs_iorcreate4(hf3fs_ior* ior, const char*, int, bool, int, int, int,
                     uint64_t) {
    *ior = {};
    ior->iorh = new Ring;
    return 0;
}
void hf3fs_iordestroy(hf3fs_ior* ior) {
    auto* ring = static_cast<Ring*>(ior->iorh);
    EXPECT_TRUE(ring->writes.empty()) << "I/O still owns shared staging bytes";
    delete ring;
}
int hf3fs_prep_io(const hf3fs_ior* ior, const hf3fs_iov*, bool read,
                  void* buffer, int, size_t, uint64_t bytes,
                  const void* userdata) {
    if (faults.fail_prepare) return -EIO;
    auto& ring = *static_cast<Ring*>(ior->iorh);
    const int index = ring.next_index++;
    ring.writes.push_back(
        {index, buffer, static_cast<size_t>(bytes), userdata, read});
    return index;
}
int hf3fs_submit_ios(const hf3fs_ior*) {
    return std::exchange(faults.submit_error, 0);
}
int hf3fs_wait_for_ios(const hf3fs_ior* ior, hf3fs_cqe* cqe, int, int,
                       const timespec*) {
    {
        std::unique_lock lock(wait_mutex);
        if (block_wait) {
            wait_entered = true;
            wait_cv.notify_all();
            wait_cv.wait(lock, [] { return wait_released; });
        }
    }
    if (!faults.waits.empty()) {
        const int result = faults.waits.front();
        faults.waits.pop_front();
        return result;
    }
    if (std::exchange(faults.unexpected_completion, false)) {
        *cqe = {-999, 0, 0, nullptr};
        return 1;
    }
    auto& ring = *static_cast<Ring*>(ior->iorh);
    if (ring.writes.empty()) return 0;
    const auto write = ring.writes.front();
    ring.writes.pop_front();
    ++faults.completed;
    if (write.read) std::memset(write.buffer, 'R', write.bytes);
    faults.data.assign(static_cast<char*>(write.buffer),
                       static_cast<char*>(write.buffer) + write.bytes);
    *cqe = {
        write.index, 0,
        faults.result == -1 ? static_cast<int64_t>(write.bytes) : faults.result,
        write.userdata};
    return 1;
}
}  // extern "C"

namespace mooncake::test {
class Hf3fsWriteLifetimeTest : public ::testing::Test {
   protected:
    void SetUp() override { faults = {}; }
    tl::expected<size_t, ErrorCode> Write() {
        Hf3fsAdapter adapter;
        EXPECT_TRUE(adapter.Init("/"));
        std::vector<char> source(4096, 'W');
        iovec iov{source.data(), source.size()};
        const auto result = adapter.WriteAt(10, &iov, 1, 0);
        EXPECT_TRUE(adapter.Shutdown());
        return result;
    }
};
TEST_F(Hf3fsWriteLifetimeTest, CompleteWritePreservesData) {
    const auto result = Write();
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 4096);
    EXPECT_EQ(faults.data, std::vector<char>(4096, 'W'));
}
TEST_F(Hf3fsWriteLifetimeTest, FailedSubmitNotificationStillDrains) {
    faults.submit_error = -EIO;
    EXPECT_FALSE(Write());
    EXPECT_EQ(faults.completed, 1);
}
TEST_F(Hf3fsWriteLifetimeTest, InterruptedAndEmptyWaitsRetainOwnership) {
    faults.waits = {-EINTR, -EAGAIN, 0};
    EXPECT_TRUE(Write());
    EXPECT_EQ(faults.completed, 1);
}
TEST_F(Hf3fsWriteLifetimeTest, FatalWaitDoesNotReleasePendingWrite) {
    faults.waits = {-EIO};
    EXPECT_FALSE(Write());
    EXPECT_EQ(faults.completed, 1);
}
TEST_F(Hf3fsWriteLifetimeTest, UnexpectedCompletionDoesNotReplacePendingWrite) {
    faults.unexpected_completion = true;
    EXPECT_FALSE(Write());
    EXPECT_EQ(faults.completed, 1);
}
TEST_F(Hf3fsWriteLifetimeTest, ShortZeroAndOversizedCompletionsFail) {
    for (const int64_t result :
         {int64_t{0}, int64_t{4095}, int64_t{4097}, int64_t{-EIO}}) {
        faults = {};
        faults.result = result;
        EXPECT_FALSE(Write());
        EXPECT_EQ(faults.completed, 1);
    }
}
TEST_F(Hf3fsWriteLifetimeTest, PrepareFailureHasNothingToDrain) {
    faults.fail_prepare = true;
    EXPECT_FALSE(Write());
    EXPECT_EQ(faults.completed, 0);
}
TEST_F(Hf3fsWriteLifetimeTest,
       FailedReadDrainsBeforeTheSharedIovIsReusedForWrite) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<char> destination(4096, '?');
    iovec read{destination.data(), destination.size()};
    faults.submit_error = -EIO;
    EXPECT_FALSE(adapter.ReadAt(10, &read, 1, 0));
    EXPECT_EQ(faults.completed, 1);
    std::vector<char> source(4096, 'W');
    iovec write{source.data(), source.size()};
    ASSERT_TRUE(adapter.WriteAt(10, &write, 1, 0));
    EXPECT_EQ(faults.data, source);
    EXPECT_EQ(faults.completed, 2);
    EXPECT_TRUE(adapter.Shutdown());
}

TEST_F(Hf3fsWriteLifetimeTest, ShutdownWaitsForThePreparedIoToFinish) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<char> source(4096, 'W');
    iovec iov{source.data(), source.size()};
    {
        std::lock_guard lock(wait_mutex);
        block_wait = true;
        wait_entered = wait_released = false;
    }
    auto written = std::async(std::launch::async,
                              [&] { return adapter.WriteAt(10, &iov, 1, 0); });
    {
        std::unique_lock lock(wait_mutex);
        EXPECT_TRUE(wait_cv.wait_for(lock, std::chrono::seconds(1),
                                     [] { return wait_entered; }));
    }
    auto stopped =
        std::async(std::launch::async, [&] { return adapter.Shutdown(); });
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(20)),
              std::future_status::timeout);
    {
        std::lock_guard lock(wait_mutex);
        wait_released = true;
        block_wait = false;
        wait_cv.notify_all();
    }
    EXPECT_TRUE(written.get());
    EXPECT_TRUE(stopped.get());
}

}  // namespace mooncake::test
