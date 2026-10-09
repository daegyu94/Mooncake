#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>
#include "hf3fs/hf3fs.h"
#include "storage/distributed/hf3fs_adapter.h"

// The real adapter/resource manager is linked without a 3FS service. Any
// accidental USRBIO access fails; metadata syscalls use real temporary files.
extern "C" {
int hf3fs_extract_mount_point(char*, int, const char*) { return -1; }
int hf3fs_reg_fd(int, uint64_t) { return -ENOSYS; }
void hf3fs_dereg_fd(int) {}
int hf3fs_iovcreate(hf3fs_iov*, const char*, size_t, size_t, int) {
    return -ENOSYS;
}
void hf3fs_iovdestroy(hf3fs_iov*) {}
int hf3fs_iorcreate4(hf3fs_ior*, const char*, int, bool, int, int, int,
                     uint64_t) {
    return -ENOSYS;
}
void hf3fs_iordestroy(hf3fs_ior*) {}
int hf3fs_prep_io(const hf3fs_ior*, const hf3fs_iov*, bool, void*, int, size_t,
                  uint64_t, const void*) {
    return -ENOSYS;
}
int hf3fs_submit_ios(const hf3fs_ior*) { return -ENOSYS; }
int hf3fs_wait_for_ios(const hf3fs_ior*, hf3fs_cqe*, int, int,
                       const timespec*) {
    return -ENOSYS;
}

int __real_fallocate(int, int, off_t, off_t);
int __real_ftruncate(int, off_t);
int __real_close(int);
}
namespace {
int allocation_error = 0, truncate_error = 0;
}
extern "C" int __wrap_fallocate(int fd, int mode, off_t offset, off_t length) {
    if (allocation_error) {
        errno = allocation_error;
        return -1;
    }
    return __real_fallocate(fd, mode, offset, length);
}
extern "C" int __wrap_ftruncate(int fd, off_t length) {
    if (truncate_error) {
        errno = truncate_error;
        return -1;
    }
    return __real_ftruncate(fd, length);
}
extern "C" int __wrap_close(int fd) {
    int result = __real_close(fd);
    errno = EBUSY;  // Cleanup must not hide the original failure.
    return result;
}

namespace mooncake::test {
class Hf3fsPreallocationTest : public ::testing::Test {
   protected:
    void SetUp() override {
        char name[] = "/tmp/hf3fs-prealloc-XXXXXX";
        auto* directory = ::mkdtemp(name);
        ASSERT_NE(directory, nullptr);
        root = directory;
        path = (root / "file").string();
        const char* value = std::getenv("MOONCAKE_DFS_PREALLOCATION_MODE");
        had_mode = value != nullptr;
        if (value) old_mode = value;
        ::unsetenv("MOONCAKE_DFS_PREALLOCATION_MODE");
        allocation_error = truncate_error = 0;
    }
    void TearDown() override {
        if (had_mode)
            ::setenv("MOONCAKE_DFS_PREALLOCATION_MODE", old_mode.c_str(), 1);
        else
            ::unsetenv("MOONCAKE_DFS_PREALLOCATION_MODE");
        std::filesystem::remove_all(root);
    }
    std::filesystem::path root;
    std::string path, old_mode;
    bool had_mode = false;
};
TEST_F(Hf3fsPreallocationTest, SpaceExhaustionIsNeverSparseSuccess) {
    for (const char* mode : {"best_effort", "strict"}) {
        ::setenv("MOONCAKE_DFS_PREALLOCATION_MODE", mode, 1);
        Hf3fsAdapter adapter;
        ASSERT_TRUE(adapter.Init(root.string()));
        allocation_error = ENOSPC;
        auto result = adapter.PreallocateFile(path, 4096);
        ASSERT_FALSE(result) << mode;
        EXPECT_EQ(result.error(), ErrorCode::FILE_WRITE_FAIL);
        EXPECT_EQ(errno, ENOSPC);
        EXPECT_EQ(std::filesystem::file_size(path), 0);
    }
}
TEST_F(Hf3fsPreallocationTest, StrictRejectsUnsupportedReservation) {
    ::setenv("MOONCAKE_DFS_PREALLOCATION_MODE", "strict", 1);
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init(root.string()));
    allocation_error = EOPNOTSUPP;
    EXPECT_FALSE(adapter.PreallocateFile(path, 4096));
    EXPECT_EQ(std::filesystem::file_size(path), 0);
}
TEST_F(Hf3fsPreallocationTest, BestEffortAllowsOnlyUnsupportedSparseFallback) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init(root.string()));
    allocation_error = EOPNOTSUPP;
    ASSERT_TRUE(adapter.PreallocateFile(path, 4096));
    EXPECT_EQ(std::filesystem::file_size(path), 4096);
    truncate_error = EIO;
    auto failed = adapter.PreallocateFile(path, 8192);
    EXPECT_FALSE(failed);
    EXPECT_EQ(errno, EIO);
    EXPECT_EQ(std::filesystem::file_size(path), 4096);
}
TEST_F(Hf3fsPreallocationTest, IoFailureIsNotSparseSuccess) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init(root.string()));
    allocation_error = EIO;
    EXPECT_FALSE(adapter.PreallocateFile(path, 4096));
    EXPECT_EQ(std::filesystem::file_size(path), 0);
}
TEST_F(Hf3fsPreallocationTest, InvalidPolicyFailsInitialization) {
    ::setenv("MOONCAKE_DFS_PREALLOCATION_MODE", "strcit", 1);
    Hf3fsAdapter adapter;
    auto result = adapter.Init(root.string());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
}
TEST_F(Hf3fsPreallocationTest, SuccessfulAllocationAndInvalidSize) {
    Hf3fsAdapter adapter;
    ASSERT_TRUE(adapter.Init(root.string()));
    ASSERT_TRUE(adapter.PreallocateFile(path, 4096));
    EXPECT_EQ(std::filesystem::file_size(path), 4096);
    auto result =
        adapter.PreallocateFile(path, std::numeric_limits<uint64_t>::max());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(std::filesystem::file_size(path), 4096);
}
}  // namespace mooncake::test
