#include "storage/distributed/posix_fs_adapter.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>

namespace mooncake {

namespace {

tl::expected<void, ErrorCode> ValidateIov(const iovec* iov, int iovcnt) {
    if (iovcnt < 0 || (iovcnt > 0 && iov == nullptr)) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    for (int i = 0; i < iovcnt; ++i) {
        if (!iov[i].iov_base && iov[i].iov_len > 0) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }
    return {};
}

ErrorCode ReadOpenError() {
    return errno == ENOENT ? ErrorCode::FILE_NOT_FOUND
                           : ErrorCode::FILE_OPEN_FAIL;
}

// Whether the filesystem, bypassing the kernel's cached attributes, reports
// the file as extending past `end`.
bool FileExtendsPast(int fd, uint64_t end) {
#if defined(__linux__)
    struct statx info{};
    return ::statx(fd, "", AT_EMPTY_PATH | AT_STATX_FORCE_SYNC, STATX_SIZE,
                   &info) == 0 &&
           (info.stx_mask & STATX_SIZE) != 0 && info.stx_size > end;
#else
    (void)fd;
    (void)end;
    return false;
#endif
}

}  // namespace

tl::expected<void, ErrorCode> PosixFsAdapter::Init(
    const std::string& mount_path) {
    mount_path_ = mount_path;
    std::error_code ec;
    std::filesystem::create_directories(mount_path_, ec);
    if (ec) return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    return {};
}

tl::expected<void, ErrorCode> PosixFsAdapter::Shutdown() { return {}; }

tl::expected<void, ErrorCode> PosixFsAdapter::DeleteFile(
    const std::string& path) {
    if (::unlink(path.c_str()) != 0) {
        if (errno == ENOENT)
            return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

tl::expected<bool, ErrorCode> PosixFsAdapter::FileExists(
    const std::string& path) {
    if (::access(path.c_str(), F_OK) == 0) return true;
    if (errno == ENOENT) return false;
    return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
}

tl::expected<std::vector<std::string>, ErrorCode> PosixFsAdapter::ListFiles(
    const std::string& dir) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) {
        if (errno == ENOENT)
            return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }

    std::vector<std::string> result;
    while (auto* entry = ::readdir(d)) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (entry->d_type == DT_DIR) continue;
        if (entry->d_type == DT_UNKNOWN) {
            struct stat st;
            if (::stat((dir + "/" + name).c_str(), &st) == 0 &&
                S_ISDIR(st.st_mode)) {
                continue;
            }
        }
        result.push_back(std::move(name));
    }
    ::closedir(d);
    return result;
}

tl::expected<int, ErrorCode> PosixFsAdapter::OpenFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    return fd;
}

tl::expected<int, ErrorCode> PosixFsAdapter::OpenExistingFile(
    const std::string& path) {
    int fd;
    do {
        fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) return tl::make_unexpected(ReadOpenError());
    return fd;
}

tl::expected<void, ErrorCode> PosixFsAdapter::CloseFile(int fd) {
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_INVALID_HANDLE);
    if (::close(fd) != 0) {
        return tl::make_unexpected(ErrorCode::FILE_INVALID_HANDLE);
    }
    return {};
}

tl::expected<void, ErrorCode> PosixFsAdapter::PreallocateFile(
    const std::string& path, uint64_t size) {
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);

    int rc = ::ftruncate(fd, static_cast<off_t>(size));
    int saved_errno = errno;
    ::close(fd);
    errno = saved_errno;
    if (rc != 0) return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    return {};
}

tl::expected<size_t, ErrorCode> PosixFsAdapter::WriteAt(int fd,
                                                        const iovec* iov,
                                                        int iovcnt,
                                                        int64_t offset) {
    if (fd < 0 || offset < 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto valid = ValidateIov(iov, iovcnt);
    if (!valid) return tl::make_unexpected(valid.error());
    if (iovcnt == 0) return size_t{0};

    ssize_t ret;
    do {
        ret = ::pwritev(fd, iov, iovcnt, static_cast<off_t>(offset));
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    return static_cast<size_t>(ret);
}

tl::expected<size_t, ErrorCode> PosixFsAdapter::ReadAt(int fd, iovec* iov,
                                                       int iovcnt,
                                                       int64_t offset) {
    if (fd < 0 || offset < 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto valid = ValidateIov(iov, iovcnt);
    if (!valid) return tl::make_unexpected(valid.error());
    if (iovcnt == 0) return size_t{0};

    size_t requested = 0;
    for (int i = 0; i < iovcnt; ++i) requested += iov[i].iov_len;

    ssize_t ret;
    do {
        ret = ::preadv(fd, iov, iovcnt, static_cast<off_t>(offset));
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);

    // A FUSE filesystem that answers a read over a sparse hole with a short
    // read (e.g. 3FS) makes the kernel shrink its cached file size, so reads
    // of written data past the hole end early. Refresh the size from the
    // filesystem and retry before reporting a short read. Other readers of
    // the file can shrink the size again, so retry a bounded number of times.
    constexpr int kMaxStaleSizeRetries = 3;
    for (int attempt = 0;
         attempt < kMaxStaleSizeRetries &&
         static_cast<size_t>(ret) < requested &&
         FileExtendsPast(fd, static_cast<uint64_t>(offset) + ret);
         ++attempt) {
        do {
            ret = ::preadv(fd, iov, iovcnt, static_cast<off_t>(offset));
        } while (ret < 0 && errno == EINTR);
        if (ret < 0) return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }
    return static_cast<size_t>(ret);
}

}  // namespace mooncake
