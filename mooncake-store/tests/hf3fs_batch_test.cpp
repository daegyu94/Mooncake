#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <unordered_map>
#include <vector>

#include "hf3fs/hf3fs.h"
#include "storage/distributed/hf3fs_adapter.h"

// Link the production adapter/resource manager to this C API double, without
// the native 3FS library. I/O consumes staging buffers only when reaped, and
// CQEs arrive in reverse order, making buffer aliasing and premature reuse
// observable. These are component tests, not native filesystem tests.
namespace {
struct Io {
    int index, fd;
    size_t offset, bytes;
    void* buffer;
    const void* userdata;
    bool read;
};
struct Ring {
    std::deque<Io> pending;
    int next = 0;
};
struct Fake {
    std::unordered_map<int, std::vector<char>> files;
    int prep = 0, submit = 0, wait = 0;
    int fail_prep = -1, fail_cqe = -1;
    int64_t bad_result = -EIO;
    int submit_error = 0, wait_error = 0;
    size_t max_count = 0, max_bytes = 0;
    bool malformed = false, wrong_userdata = false;
} fake;
}  // namespace
extern "C" {
int hf3fs_extract_mount_point(char*, int, const char*) { return -1; }
int hf3fs_reg_fd(int, uint64_t) { return 0; }
void hf3fs_dereg_fd(int) {}
int hf3fs_iovcreate(hf3fs_iov* iov, const char*, size_t bytes, size_t, int) {
    *iov = {};
    if (::posix_memalign(reinterpret_cast<void**>(&iov->base), 4096, bytes))
        return -ENOMEM;
    iov->size = bytes;
    return 0;
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
    EXPECT_TRUE(ring->pending.empty()) << "destroyed ring before drain";
    delete ring;
}
int hf3fs_prep_io(const hf3fs_ior* ior, const hf3fs_iov* iov, bool read,
                  void* buffer, int fd, size_t offset, uint64_t bytes,
                  const void* userdata) {
    if (++fake.prep == fake.fail_prep) return -EIO;
    auto* ring = static_cast<Ring*>(ior->iorh);
    auto* begin = static_cast<uint8_t*>(buffer);
    EXPECT_GE(begin, iov->base);
    EXPECT_LE(begin + bytes, iov->base + iov->size);
    size_t inflight = bytes;
    for (const auto& other : ring->pending) {
        auto* prior = static_cast<uint8_t*>(other.buffer);
        EXPECT_FALSE(begin < prior + other.bytes && prior < begin + bytes)
            << "live staging ranges overlap";
        inflight += other.bytes;
    }
    int index = ring->next++;
    ring->pending.push_back({index, fd, offset, static_cast<size_t>(bytes),
                             buffer, userdata, read});
    fake.max_count = std::max(fake.max_count, ring->pending.size());
    fake.max_bytes = std::max(fake.max_bytes, inflight);
    return index;
}
int hf3fs_submit_ios(const hf3fs_ior*) {
    ++fake.submit;
    return std::exchange(fake.submit_error, 0);
}
int hf3fs_wait_for_ios(const hf3fs_ior* ior, hf3fs_cqe* cqes, int capacity, int,
                       const timespec*) {
    ++fake.wait;
    if (fake.wait_error) return std::exchange(fake.wait_error, 0);
    if (fake.malformed) {
        fake.malformed = false;
        cqes[0] = {-999, 0, 0, nullptr};
        return 1;
    }
    auto* ring = static_cast<Ring*>(ior->iorh);
    int count = 0;
    while (!ring->pending.empty() && count < std::min(capacity, 2)) {
        const auto io = ring->pending.back();
        ring->pending.pop_back();
        auto& file = fake.files[io.fd];
        if (file.size() < io.offset + io.bytes)
            file.resize(io.offset + io.bytes);
        int64_t result = io.index == fake.fail_cqe ? fake.bad_result : io.bytes;
        if (result == static_cast<int64_t>(io.bytes)) {
            if (io.read)
                std::memcpy(io.buffer, file.data() + io.offset, io.bytes);
            else
                std::memcpy(file.data() + io.offset, io.buffer, io.bytes);
        }
        cqes[count++] = {io.index, 0, result, io.userdata};
        if (fake.wrong_userdata) {
            fake.wrong_userdata = false;
            cqes[count - 1].userdata = nullptr;
        }
    }
    return count;
}
}
namespace mooncake::test {
class Hf3fsBatchTest : public ::testing::Test {
   protected:
    void SetUp() override { fake = {}; }
    std::vector<FileIoRequest> Requests(std::vector<std::vector<char>>& data) {
        std::vector<FileIoRequest> requests;
        for (size_t i = 0; i < data.size(); ++i)
            requests.push_back({10,
                                {{data[i].data(), data[i].size()}},
                                static_cast<int64_t>(i * 16384)});
        return requests;
    }
};
TEST_F(Hf3fsBatchTest, BoundedSlotsAndReversePartialCompletionsPreserveData) {
    Hf3fsAdapter adapter({3, 12288});
    ASSERT_TRUE(adapter.Init("/").has_value());
    std::vector<std::vector<char>> writes(8, std::vector<char>(6000));
    for (size_t i = 0; i < writes.size(); ++i)
        std::fill(writes[i].begin(), writes[i].end(), 'A' + i);
    auto requests = Requests(writes);
    auto results = adapter.BatchWriteAt(requests);
    for (const auto& result : results) {
        ASSERT_TRUE(result);
        EXPECT_EQ(*result, 6000U);
    }
    std::vector<std::vector<char>> reads(8, std::vector<char>(6000));
    requests = Requests(reads);
    results = adapter.BatchReadAt(requests);
    for (const auto& result : results) ASSERT_TRUE(result);
    EXPECT_EQ(reads, writes);
    EXPECT_LE(fake.max_count, 3U);
    EXPECT_LE(fake.max_bytes, 12288U);
    EXPECT_GT(fake.max_count, 1U);
}
TEST_F(Hf3fsBatchTest, OneLargeObjectCrossesWindowsAndIovecBoundaries) {
    Hf3fsAdapter adapter({4, 8192});
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<char> a(9001, 'a'), b(8003, 'b'), ra(a.size()), rb(b.size());
    auto results = adapter.BatchWriteAt(
        {{10,
          {{nullptr, 0}, {a.data(), a.size()}, {b.data(), b.size()}},
          101}});
    ASSERT_TRUE(results[0]);
    EXPECT_EQ(*results[0], a.size() + b.size());
    std::vector<FileIoRequest> reads{
        {10,
         {{ra.data(), ra.size()}, {nullptr, 0}, {rb.data(), rb.size()}},
         101}};
    results = adapter.BatchReadAt(reads);
    ASSERT_TRUE(results[0]);
    EXPECT_EQ(a, ra);
    EXPECT_EQ(b, rb);
    EXPECT_LE(fake.max_bytes, 8192U);
}
TEST_F(Hf3fsBatchTest, PrepareFailureDrainsEarlierSlotsAndKeepsOtherResults) {
    Hf3fsAdapter adapter({4, 16384});
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<std::vector<char>> data(4, std::vector<char>(4096, 'p'));
    fake.fail_prep = 2;
    auto results = adapter.BatchWriteAt(Requests(data));
    EXPECT_TRUE(results[0]);
    ASSERT_FALSE(results[1]);
    EXPECT_EQ(results[1].error(), ErrorCode::FILE_WRITE_FAIL);
    EXPECT_TRUE(results[2]);
    EXPECT_TRUE(results[3]);
}
TEST_F(Hf3fsBatchTest, NegativeShortAndOversizedCompletionFailOnlyTheirObject) {
    for (int64_t bad : {-EIO, 4095, 4097}) {
        fake = {};
        Hf3fsAdapter adapter;
        ASSERT_TRUE(adapter.Init("/"));
        std::vector<std::vector<char>> data(3, std::vector<char>(4096, 'c'));
        fake.fail_cqe = 1;
        fake.bad_result = bad;
        auto requests = Requests(data);
        auto results = adapter.BatchReadAt(requests);
        EXPECT_TRUE(results[0]);
        ASSERT_FALSE(results[1]);
        EXPECT_EQ(results[1].error(), ErrorCode::FILE_READ_FAIL);
        EXPECT_TRUE(results[2]);
        EXPECT_EQ(data[1], std::vector<char>(4096, 'c'))
            << "failed CQE must not scatter";
    }
}
TEST_F(Hf3fsBatchTest, FailedSubmitStillDrainsPreparedBuffers) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<std::vector<char>> data(3, std::vector<char>(4096, 'd'));
    fake.submit_error = -EIO;
    auto results = adapter.BatchWriteAt(Requests(data));
    for (const auto& result : results) EXPECT_FALSE(result);
    EXPECT_EQ(fake.files[10][0], 'd');
    EXPECT_EQ(fake.submit, 1);
}
TEST_F(Hf3fsBatchTest, InterruptedOrFailedWaitDrainsBeforeReturn) {
    for (int error : {-EINTR, -EIO}) {
        fake = {};
        Hf3fsAdapter adapter;
        ASSERT_TRUE(adapter.Init("/"));
        std::vector<std::vector<char>> data(3, std::vector<char>(4096, 'w'));
        fake.wait_error = error;
        auto results = adapter.BatchWriteAt(Requests(data));
        for (const auto& result : results)
            EXPECT_EQ(result.has_value(), error == -EINTR);
        EXPECT_GE(fake.wait, 3);
    }
}
TEST_F(Hf3fsBatchTest,
       MalformedCompletionFencesRingAfterDrainingValidCompletions) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<std::vector<char>> data(3, std::vector<char>(4096, 'f'));
    fake.malformed = true;
    auto results = adapter.BatchWriteAt(Requests(data));
    for (const auto& result : results) EXPECT_FALSE(result);
    int prepared = fake.prep;
    results = adapter.BatchWriteAt(Requests(data));
    EXPECT_EQ(fake.prep, prepared);
    for (const auto& result : results) EXPECT_FALSE(result);
}
TEST_F(Hf3fsBatchTest, OverlappingWritesKeepInputOrder) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<char> first(8192, 'a'), last(4096, 'b');
    auto results =
        adapter.BatchWriteAt({{10, {{first.data(), first.size()}}, 0},
                              {10, {{last.data(), last.size()}}, 4096}});
    for (const auto& result : results) ASSERT_TRUE(result);
    EXPECT_EQ(fake.submit, 2);
    EXPECT_EQ(std::string(fake.files[10].data(), 4096), std::string(4096, 'a'));
    EXPECT_EQ(std::string(fake.files[10].data() + 4096, 4096),
              std::string(4096, 'b'));
}
TEST_F(Hf3fsBatchTest, WrongUserdataFailsAndDrainsUsingRingIndex) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    std::vector<std::vector<char>> data(3, std::vector<char>(4096, 'f'));
    fake.wrong_userdata = true;
    auto results = adapter.BatchWriteAt(Requests(data));
    for (const auto& result : results) EXPECT_FALSE(result);
    const int prepared = fake.prep;
    adapter.BatchWriteAt(Requests(data));
    EXPECT_EQ(fake.prep, prepared);
}
TEST_F(Hf3fsBatchTest, InvalidObjectDoesNotShiftResultsOrTouchNativeApi) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    char value = 'v';
    std::vector<FileIoRequest> requests{
        {-1, {{&value, 1}}, 0},
        {10, {{nullptr, 1}}, 0},
        {10, {{&value, 2}}, std::numeric_limits<int64_t>::max()},
        {10, {{&value, 1}}, 0},
        {10, {}, 0}};
    auto results = adapter.BatchReadAt(requests);
    ASSERT_EQ(results.size(), requests.size());
    for (int i = 0; i < 3; ++i) {
        ASSERT_FALSE(results[i]);
        EXPECT_EQ(results[i].error(), ErrorCode::INVALID_PARAMS);
    }
    ASSERT_TRUE(results[3]);
    EXPECT_EQ(*results[3], 1U);
    ASSERT_TRUE(results[4]);
    EXPECT_EQ(*results[4], 0U);
    EXPECT_EQ(fake.prep, 1);
}
TEST_F(Hf3fsBatchTest, AliasedReadDestinationsRetainInputOrder) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init("/"));
    fake.files[10] = std::vector<char>(8192, 'a');
    std::fill(fake.files[10].begin() + 4096, fake.files[10].end(), 'b');
    std::vector<char> output(4096);
    std::vector<FileIoRequest> requests{
        {10, {{output.data(), output.size()}}, 0},
        {10, {{output.data(), output.size()}}, 4096}};
    auto results = adapter.BatchReadAt(requests);
    for (const auto& result : results) ASSERT_TRUE(result);
    EXPECT_EQ(output, std::vector<char>(4096, 'b'));
}
TEST_F(Hf3fsBatchTest, RejectsZeroBoundsAndMalformedEnvironment) {
    Hf3fsAdapter no_depth({0, 4096});
    EXPECT_FALSE(no_depth.Init("/"));
    Hf3fsAdapter no_bytes({1, 0});
    EXPECT_FALSE(no_bytes.Init("/"));
    const char* existing = std::getenv("MOONCAKE_DFS_IO_BATCH_SIZE");
    std::string saved = existing ? existing : "";
    ::setenv("MOONCAKE_DFS_IO_BATCH_SIZE", "2x", 1);
    Hf3fsAdapter adapter;
    EXPECT_FALSE(adapter.Init("/"));
    if (existing)
        ::setenv("MOONCAKE_DFS_IO_BATCH_SIZE", saved.c_str(), 1);
    else
        ::unsetenv("MOONCAKE_DFS_IO_BATCH_SIZE");
}
}  // namespace mooncake::test
