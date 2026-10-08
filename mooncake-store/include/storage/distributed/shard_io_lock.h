#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace mooncake {

// Ranges are half-open, validated allocation ranges. Readers may overlap;
// a writer excludes overlapping reads. Shard writes remain serialized to
// retain protection of filesystem write metadata beyond the byte range.
class ShardIoLock {
   public:
    struct Range {
        uint64_t begin;
        uint64_t end;
        bool write;
    };

    class Guard {
       public:
        Guard(Guard&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)),
              ranges_(std::move(other.ranges_)) {}
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        ~Guard() {
            if (owner_) owner_->Release(ranges_);
        }

       private:
        friend class ShardIoLock;
        Guard(ShardIoLock* owner,
              std::shared_ptr<const std::vector<Range>> ranges)
            : owner_(owner), ranges_(std::move(ranges)) {}
        ShardIoLock* owner_;
        std::shared_ptr<const std::vector<Range>> ranges_;
    };

    Guard Lock(std::vector<Range> ranges) {
        auto owned =
            std::make_shared<const std::vector<Range>>(std::move(ranges));
        std::unique_lock lock(mutex_);
        waiting_.push_back(owned);
        ready_.wait(lock, [&] {
            for (const auto& active : active_) {
                if (Conflicts(*active, *owned)) return false;
            }
            // Do not let new overlapping readers starve a queued writer.
            for (const auto& earlier : waiting_) {
                if (earlier == owned) break;
                if (Conflicts(*earlier, *owned)) return false;
            }
            return true;
        });
        std::erase(waiting_, owned);
        active_.push_back(owned);
        return Guard(this, std::move(owned));
    }

   private:
    static bool Conflicts(const std::vector<Range>& left,
                          const std::vector<Range>& right) {
        for (const auto& a : left) {
            for (const auto& b : right) {
                if ((a.write && b.write) ||
                    ((a.write || b.write) && a.begin < b.end &&
                     b.begin < a.end))
                    return true;
            }
        }
        return false;
    }
    void Release(const std::shared_ptr<const std::vector<Range>>& ranges) {
        {
            std::lock_guard lock(mutex_);
            std::erase(active_, ranges);
        }
        ready_.notify_all();
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<std::shared_ptr<const std::vector<Range>>> active_;
    std::vector<std::shared_ptr<const std::vector<Range>>> waiting_;
};

}  // namespace mooncake
