#pragma once

#include <memory>

#include "storage/distributed/fs_adapter.h"

namespace mooncake {

struct Hf3fsBatchOptions {
    size_t max_requests = 16;
    size_t max_bytes = 32ULL << 20;
};

// Forward declaration: avoid including hf3fs/hf3fs.h in the header.
// Full type is only used in hf3fs_adapter.cpp.
class USRBIOResourceManager;

class Hf3fsAdapter : public FileSystemAdapter {
   public:
    explicit Hf3fsAdapter(Hf3fsBatchOptions options = {});
    ~Hf3fsAdapter() override;  // defined in .cpp (needs complete type for
                               // unique_ptr deleter)

    tl::expected<size_t, ErrorCode> WriteFile(
        const std::string& path, std::span<const char> data) override;

    tl::expected<size_t, ErrorCode> ReadFile(const std::string& path, void* buf,
                                             size_t len) override;

    tl::expected<size_t, ErrorCode> VectorWriteFile(const std::string& path,
                                                    const iovec* iov,
                                                    int iovcnt,
                                                    off_t offset) override;

    tl::expected<size_t, ErrorCode> VectorReadFile(const std::string& path,
                                                   const iovec* iov, int iovcnt,
                                                   off_t offset) override;

    tl::expected<void, ErrorCode> DeleteFile(const std::string& path) override;

    tl::expected<bool, ErrorCode> FileExists(const std::string& path) override;

    tl::expected<std::vector<std::string>, ErrorCode> ListFiles(
        const std::string& dir) override;

    tl::expected<int, ErrorCode> OpenFile(const std::string& path) override;

    tl::expected<void, ErrorCode> CloseFile(int fd) override;

    tl::expected<void, ErrorCode> PreallocateFile(const std::string& path,
                                                  uint64_t size) override;

    tl::expected<size_t, ErrorCode> WriteAt(int fd, const iovec* iov,
                                            int iovcnt,
                                            int64_t offset) override;

    tl::expected<size_t, ErrorCode> ReadAt(int fd, iovec* iov, int iovcnt,
                                           int64_t offset) override;

    std::vector<tl::expected<size_t, ErrorCode>> BatchWriteAt(
        const std::vector<FileIoRequest>& requests) override;
    std::vector<tl::expected<size_t, ErrorCode>> BatchReadAt(
        std::vector<FileIoRequest>& requests) override;

    tl::expected<void, ErrorCode> Init(const std::string& mount_path) override;

    tl::expected<void, ErrorCode> Shutdown() override;

    const char* GetName() const override { return "hf3fs"; }

   private:
    std::vector<tl::expected<size_t, ErrorCode>> BatchAt(
        const std::vector<FileIoRequest>& requests, bool read);
    Hf3fsBatchOptions batch_options_;
    std::unique_ptr<USRBIOResourceManager> resource_manager_;
};

}  // namespace mooncake
