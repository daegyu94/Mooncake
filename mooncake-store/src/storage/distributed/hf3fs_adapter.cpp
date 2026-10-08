#include "storage/distributed/hf3fs_adapter.h"

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <thread>

#include <glog/logging.h>

#include "hf3fs/hf3fs.h"

namespace mooncake {

// Destructor must be defined in .cpp where USRBIOResourceManager is complete.
Hf3fsAdapter::~Hf3fsAdapter() = default;

Hf3fsAdapter::Hf3fsAdapter(Hf3fsBatchOptions options)
    : batch_options_(options) {}

tl::expected<void, ErrorCode> Hf3fsAdapter::Init(
    const std::string& mount_path) {
    for (auto [name, value] :
         {std::pair{"MOONCAKE_DFS_IO_BATCH_SIZE", &batch_options_.max_requests},
          std::pair{"MOONCAKE_DFS_IO_BATCH_BYTES",
                    &batch_options_.max_bytes}}) {
        if (const char* text = std::getenv(name)) {
            auto end = text + std::strlen(text);
            auto parsed = std::from_chars(text, end, *value);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }
    if (batch_options_.max_requests == 0 || batch_options_.max_bytes < 4096)
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    resource_manager_ = std::make_unique<USRBIOResourceManager>();
    Hf3fsConfig config{};
    char hf3fs_mount_point[PATH_MAX] = {};
    int ret = hf3fs_extract_mount_point(
        hf3fs_mount_point, sizeof(hf3fs_mount_point), mount_path.c_str());
    if (ret > 0 && ret <= static_cast<int>(sizeof(hf3fs_mount_point))) {
        config.mount_root = hf3fs_mount_point;
    } else {
        config.mount_root = mount_path;
    }
    resource_manager_->setDefaultParams(config);
    return {};
}

tl::expected<void, ErrorCode> Hf3fsAdapter::Shutdown() {
    resource_manager_.reset();
    return {};
}

tl::expected<size_t, ErrorCode> Hf3fsAdapter::WriteFile(
    const std::string& path, std::span<const char> data) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);

    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        ::unlink(path.c_str());
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto* resource = resource_manager_->getThreadResource();
    if (!resource || !resource->initialized || resource->io_failed) {
        hf3fs_dereg_fd(fd);
        close(fd);
        ::unlink(path.c_str());
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto& threefs_iov = resource->iov_;
    auto& ior_write = resource->ior_write_;
    const char* data_ptr = data.data();
    size_t length = data.size();
    size_t total_written = 0;
    off_t offset = 0;

    while (total_written < length) {
        size_t chunk =
            std::min(length - total_written, resource->config_.iov_size);
        memcpy(threefs_iov.base, data_ptr + total_written, chunk);

        int ret = hf3fs_prep_io(&ior_write, &threefs_iov, false,
                                threefs_iov.base, fd, offset, chunk, nullptr);
        if (ret < 0) break;

        ret = hf3fs_submit_ios(&ior_write);
        if (ret < 0) break;

        struct hf3fs_cqe cqe;
        ret = hf3fs_wait_for_ios(&ior_write, &cqe, 1, 1, nullptr);
        if (ret < 0 || cqe.result < 0) break;

        size_t bytes_written = cqe.result;
        total_written += bytes_written;
        offset += bytes_written;
        if (bytes_written < chunk) break;
    }

    hf3fs_dereg_fd(fd);
    close(fd);

    if (total_written != length) {
        auto unlink_ret = ::unlink(path.c_str());
        if (unlink_ret != 0) {
            LOG(WARNING) << "Failed to clean up partial write: " << path;
        }
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return total_written;
}

tl::expected<size_t, ErrorCode> Hf3fsAdapter::ReadFile(const std::string& path,
                                                       void* buf, size_t len) {
    if (!buf && len > 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (errno == ENOENT) {
            return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
        }
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto* resource = resource_manager_->getThreadResource();
    if (!resource || !resource->initialized || resource->io_failed) {
        hf3fs_dereg_fd(fd);
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto& threefs_iov = resource->iov_;
    auto& ior_read = resource->ior_read_;
    char* dest = static_cast<char*>(buf);
    size_t total_read = 0;
    off_t offset = 0;

    while (total_read < len) {
        size_t chunk = std::min(len - total_read, resource->config_.iov_size);

        int ret = hf3fs_prep_io(&ior_read, &threefs_iov, true, threefs_iov.base,
                                fd, offset, chunk, nullptr);
        if (ret < 0) break;

        ret = hf3fs_submit_ios(&ior_read);
        if (ret < 0) break;

        struct hf3fs_cqe cqe;
        ret = hf3fs_wait_for_ios(&ior_read, &cqe, 1, 1, nullptr);
        if (ret < 0 || cqe.result < 0) break;

        size_t bytes_read = cqe.result;
        if (bytes_read == 0) break;

        memcpy(dest + total_read, threefs_iov.base, bytes_read);
        total_read += bytes_read;
        offset += bytes_read;
        if (bytes_read < chunk) break;
    }

    hf3fs_dereg_fd(fd);
    close(fd);

    if (total_read != len) {
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }
    return total_read;
}

tl::expected<size_t, ErrorCode> Hf3fsAdapter::VectorWriteFile(
    const std::string& path, const iovec* iov, int iovcnt, off_t offset) {
    for (int i = 0; i < iovcnt; ++i) {
        if (!iov[i].iov_base && iov[i].iov_len > 0) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }

    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);

    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        ::unlink(path.c_str());
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto* resource = resource_manager_->getThreadResource();
    if (!resource || !resource->initialized || resource->io_failed) {
        hf3fs_dereg_fd(fd);
        close(fd);
        ::unlink(path.c_str());
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto& threefs_iov = resource->iov_;
    auto& ior_write = resource->ior_write_;

    size_t total_length = 0;
    for (int i = 0; i < iovcnt; ++i) total_length += iov[i].iov_len;

    size_t total_written = 0;
    off_t current_offset = offset;
    size_t remaining = total_length;
    int iov_idx = 0;
    size_t iov_off = 0;

    while (remaining > 0) {
        size_t chunk = std::min(remaining, resource->config_.iov_size);

        // Copy from iovec to shared buffer
        size_t copied = 0;
        char* dest = reinterpret_cast<char*>(threefs_iov.base);
        while (copied < chunk && iov_idx < iovcnt) {
            size_t n = std::min(chunk - copied, iov[iov_idx].iov_len - iov_off);
            memcpy(dest + copied,
                   static_cast<char*>(iov[iov_idx].iov_base) + iov_off, n);
            copied += n;
            iov_off += n;
            if (iov_off >= iov[iov_idx].iov_len) {
                iov_idx++;
                iov_off = 0;
            }
        }

        int ret =
            hf3fs_prep_io(&ior_write, &threefs_iov, false, threefs_iov.base, fd,
                          current_offset, chunk, nullptr);
        if (ret < 0) break;
        ret = hf3fs_submit_ios(&ior_write);
        if (ret < 0) break;
        struct hf3fs_cqe cqe;
        ret = hf3fs_wait_for_ios(&ior_write, &cqe, 1, 1, nullptr);
        if (ret < 0 || cqe.result < 0) break;

        size_t bytes_written = cqe.result;
        total_written += bytes_written;
        current_offset += bytes_written;
        remaining -= bytes_written;
        if (bytes_written < chunk) break;
    }

    hf3fs_dereg_fd(fd);
    close(fd);

    if (total_written != total_length) {
        auto unlink_ret = ::unlink(path.c_str());
        if (unlink_ret != 0) {
            LOG(WARNING) << "Failed to clean up partial write: " << path;
        }
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return total_written;
}

tl::expected<size_t, ErrorCode> Hf3fsAdapter::VectorReadFile(
    const std::string& path, const iovec* iov, int iovcnt, off_t offset) {
    for (int i = 0; i < iovcnt; ++i) {
        if (!iov[i].iov_base && iov[i].iov_len > 0) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }

    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (errno == ENOENT) {
            return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
        }
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto* resource = resource_manager_->getThreadResource();
    if (!resource || !resource->initialized || resource->io_failed) {
        hf3fs_dereg_fd(fd);
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto& threefs_iov = resource->iov_;
    auto& ior_read = resource->ior_read_;

    size_t total_length = 0;
    for (int i = 0; i < iovcnt; ++i) total_length += iov[i].iov_len;

    size_t total_read = 0;
    off_t current_offset = offset;
    size_t remaining = total_length;
    int iov_idx = 0;
    size_t iov_off = 0;

    while (remaining > 0) {
        size_t chunk = std::min(remaining, resource->config_.iov_size);

        int ret = hf3fs_prep_io(&ior_read, &threefs_iov, true, threefs_iov.base,
                                fd, current_offset, chunk, nullptr);
        if (ret < 0) break;
        ret = hf3fs_submit_ios(&ior_read);
        if (ret < 0) break;
        struct hf3fs_cqe cqe;
        ret = hf3fs_wait_for_ios(&ior_read, &cqe, 1, 1, nullptr);
        if (ret < 0 || cqe.result < 0) break;

        size_t bytes_read = cqe.result;
        if (bytes_read == 0) break;

        // Copy from shared buffer to iovec
        size_t to_copy = bytes_read;
        char* src = reinterpret_cast<char*>(threefs_iov.base);
        while (to_copy > 0 && iov_idx < iovcnt) {
            size_t n = std::min(to_copy, iov[iov_idx].iov_len - iov_off);
            memcpy(static_cast<char*>(iov[iov_idx].iov_base) + iov_off, src, n);
            src += n;
            to_copy -= n;
            total_read += n;
            remaining -= n;
            current_offset += n;
            iov_off += n;
            if (iov_off >= iov[iov_idx].iov_len) {
                iov_idx++;
                iov_off = 0;
            }
        }
        if (bytes_read < chunk) break;
    }

    hf3fs_dereg_fd(fd);
    close(fd);

    if (total_read != total_length) {
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }
    return total_read;
}

tl::expected<void, ErrorCode> Hf3fsAdapter::DeleteFile(
    const std::string& path) {
    if (::unlink(path.c_str()) != 0) {
        if (errno == ENOENT) {
            return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
        }
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

// Note: FileExists and DeleteFile use POSIX access()/unlink() via the
// kernel VFS mount. This assumes 3FS's FUSE mount namespace is consistent
// with the USRBIO namespace used for read/write I/O.

tl::expected<bool, ErrorCode> Hf3fsAdapter::FileExists(
    const std::string& path) {
    if (::access(path.c_str(), F_OK) == 0) {
        return true;
    }
    if (errno == ENOENT) {
        return false;
    }
    return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
}

tl::expected<std::vector<std::string>, ErrorCode> Hf3fsAdapter::ListFiles(
    const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) {
        if (errno == ENOENT) {
            return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
        }
        return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }
    std::vector<std::string> result;

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (entry->d_type == DT_DIR) continue;
        if (entry->d_type == DT_UNKNOWN) {
            struct stat st;
            if (::stat((dir + "/" + name).c_str(), &st) == 0 &&
                S_ISDIR(st.st_mode))
                continue;
        }
        result.push_back(name);
    }
    closedir(d);
    return result;
}

tl::expected<int, ErrorCode> Hf3fsAdapter::OpenFile(const std::string& path) {
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);

    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    return fd;
}

tl::expected<void, ErrorCode> Hf3fsAdapter::CloseFile(int fd) {
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_INVALID_HANDLE);
    hf3fs_dereg_fd(fd);
    if (close(fd) != 0) {
        return tl::make_unexpected(ErrorCode::FILE_INVALID_HANDLE);
    }
    return {};
}

tl::expected<void, ErrorCode> Hf3fsAdapter::PreallocateFile(
    const std::string& path, uint64_t size) {
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);

    int rc = fallocate(fd, 0, 0, static_cast<off_t>(size));
    if (rc != 0) {
        rc = ftruncate(fd, static_cast<off_t>(size));
    }
    int saved_errno = errno;
    close(fd);

    if (rc != 0) {
        errno = saved_errno;
        LOG(ERROR) << "Failed to preallocate DFS file " << path
                   << ", size=" << size << ", error=" << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

namespace {
void CopyIoChunk(const std::vector<iovec>& iovs, size_t offset, char* slot,
                 size_t bytes, bool read) {
    for (const auto& iov : iovs) {
        if (offset >= iov.iov_len) {
            offset -= iov.iov_len;
            continue;
        }
        const size_t n = std::min(bytes, iov.iov_len - offset);
        char* buffer = static_cast<char*>(iov.iov_base) + offset;
        if (read)
            std::memcpy(buffer, slot, n);
        else
            std::memcpy(slot, buffer, n);
        slot += n;
        bytes -= n;
        offset = 0;
        if (bytes == 0) break;
    }
}
}  // namespace

tl::expected<size_t, ErrorCode> Hf3fsAdapter::WriteAt(int fd, const iovec* iov,
                                                      int iovcnt,
                                                      int64_t offset) {
    if (iovcnt < 0 || (iovcnt && !iov))
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    FileIoRequest request{fd, {}, offset};
    if (iovcnt) request.iovs.assign(iov, iov + iovcnt);
    return BatchWriteAt({request}).front();
}

tl::expected<size_t, ErrorCode> Hf3fsAdapter::ReadAt(int fd, iovec* iov,
                                                     int iovcnt,
                                                     int64_t offset) {
    if (iovcnt < 0 || (iovcnt && !iov))
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    FileIoRequest request{fd, {}, offset};
    if (iovcnt) request.iovs.assign(iov, iov + iovcnt);
    std::vector<FileIoRequest> requests{std::move(request)};
    return BatchReadAt(requests).front();
}

std::vector<tl::expected<size_t, ErrorCode>> Hf3fsAdapter::BatchWriteAt(
    const std::vector<FileIoRequest>& requests) {
    return BatchAt(requests, false);
}

std::vector<tl::expected<size_t, ErrorCode>> Hf3fsAdapter::BatchReadAt(
    std::vector<FileIoRequest>& requests) {
    return BatchAt(requests, true);
}

std::vector<tl::expected<size_t, ErrorCode>> Hf3fsAdapter::BatchAt(
    const std::vector<FileIoRequest>& requests, bool read) {
    const auto failure =
        read ? ErrorCode::FILE_READ_FAIL : ErrorCode::FILE_WRITE_FAIL;
    std::vector<tl::expected<size_t, ErrorCode>> results(
        requests.size(), tl::make_unexpected(failure));
    if (requests.empty()) return results;
    auto* resource =
        resource_manager_ ? resource_manager_->getThreadResource() : nullptr;
    if (!resource || !resource->initialized || resource->io_failed)
        return results;
    const size_t budget =
        std::min({batch_options_.max_bytes, resource->config_.iov_size,
                  static_cast<size_t>(resource->iov_.size)});
    const size_t depth =
        std::min(batch_options_.max_requests, resource->config_.ior_entries);
    if (budget == 0 || depth == 0 || !resource->iov_.base) return results;
    auto& ring = read ? resource->ior_read_ : resource->ior_write_;
    struct State {
        size_t size = 0, scheduled = 0, completed = 0;
        bool failed = false;
    };
    struct Slot {
        size_t object, offset, bytes, staging;
        int index = -1;
        bool pending = true;
    };
    std::vector<State> states(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& r = requests[i];
        auto& state = states[i];
        state.failed = r.fd < 0 || r.offset < 0;
        for (const auto& iov : r.iovs) {
            if ((!iov.iov_base && iov.iov_len) ||
                iov.iov_len > std::numeric_limits<size_t>::max() - state.size) {
                state.failed = true;
                break;
            }
            state.size += iov.iov_len;
        }
        if (!state.failed &&
            state.size > static_cast<uint64_t>(
                             std::numeric_limits<int64_t>::max() - r.offset))
            state.failed = true;
        if (state.failed)
            results[i] = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    size_t object = 0;
    std::vector<Slot> slots;
    slots.reserve(
        depth);  // userdata addresses stay stable until every CQE arrives.
    std::vector<hf3fs_cqe> cqes(depth);
    while (object < requests.size()) {
        slots.clear();
        size_t used = 0;
        while (object < requests.size() && slots.size() < depth &&
               used < budget) {
            auto& state = states[object];
            const auto& r = requests[object];
            if (state.failed || state.scheduled == state.size) {
                ++object;
                continue;
            }
            const size_t bytes =
                std::min(state.size - state.scheduled, budget - used);
            const uint64_t begin =
                static_cast<uint64_t>(r.offset) + state.scheduled;
            bool overlap = false;
            if (!read) {
                for (const auto& slot : slots) {
                    const auto& prior = requests[slot.object];
                    const uint64_t prior_begin = prior.offset + slot.offset;
                    overlap |= prior.fd == r.fd &&
                               begin < prior_begin + slot.bytes &&
                               prior_begin < begin + bytes;
                }
            }
            if (overlap)
                break;  // Preserve request order for overlapping writes.
            slots.push_back({object, state.scheduled, bytes, used});
            auto& slot = slots.back();
            char* buffer = reinterpret_cast<char*>(resource->iov_.base) + used;
            if (!read) CopyIoChunk(r.iovs, slot.offset, buffer, bytes, false);
            slot.index = hf3fs_prep_io(&ring, &resource->iov_, read, buffer,
                                       r.fd, begin, bytes, &slot);
            if (slot.index < 0) {
                state.failed = true;
                slots.pop_back();
                ++object;
                continue;
            }
            state.scheduled += bytes;
            used += bytes;
            // Slot starts stay page aligned without reserving a full slot per
            // key.
            if (used < budget)
                used += std::min(budget - used, (4096 - used % 4096) % 4096);
        }
        if (slots.empty()) continue;
        bool wave_failed = false;
        // Submit only notifies FUSE: prepared I/O may already be executing.
        // Even a failed notification cannot permit early buffer reuse.
        wave_failed = hf3fs_submit_ios(&ring) < 0;
        size_t pending = slots.size();
        while (pending) {
            int count =
                hf3fs_wait_for_ios(&ring, cqes.data(), cqes.size(), 1, nullptr);
            if (count <= 0) {
                wave_failed |= count < 0 && count != -EINTR;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;  // Do not recycle slots or release file ranges early.
            }
            if (static_cast<size_t>(count) > cqes.size()) {
                wave_failed = resource->io_failed = true;
                count = cqes.size();
            }
            for (int i = 0; i < count; ++i) {
                const auto& cqe = cqes[i];
                auto found = std::find_if(
                    slots.begin(), slots.end(),
                    [&](const Slot& s) { return cqe.index == s.index; });
                if (found == slots.end() || !found->pending) {
                    wave_failed = resource->io_failed = true;
                    continue;
                }
                auto& slot = *found;
                auto& state = states[slot.object];
                if (cqe.userdata != &slot) {
                    wave_failed = resource->io_failed = true;
                }
                slot.pending = false;
                --pending;
                if (cqe.result < 0 ||
                    static_cast<uint64_t>(cqe.result) != slot.bytes) {
                    state.failed = true;
                    continue;
                }
                state.completed += slot.bytes;
            }
        }
        if (wave_failed) {
            for (const auto& slot : slots) states[slot.object].failed = true;
        }
        // Scatter in input order after draining the window, so even callers
        // sharing a destination buffer retain the scalar adapter's semantics.
        if (read) {
            for (const auto& slot : slots) {
                if (!states[slot.object].failed)
                    CopyIoChunk(requests[slot.object].iovs, slot.offset,
                                reinterpret_cast<char*>(resource->iov_.base) +
                                    slot.staging,
                                slot.bytes, true);
            }
        }
        if (resource->io_failed) break;
    }
    for (size_t i = 0; i < states.size(); ++i) {
        if (!states[i].failed && states[i].completed == states[i].size)
            results[i] = states[i].size;
    }
    return results;
}

}  // namespace mooncake
