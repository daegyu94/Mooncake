#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mooncake {

// Experimental trusted-client protocol, disabled unless explicitly enabled.
// These aggregates are also the coro_rpc wire types. No per-object replica
// metadata is created for a region; content discovery still needs a key index.
enum class PolicyRegionCode : uint8_t {
    OK,
    INVALID,
    STALE,
    NOT_FOUND,
    BUSY,
    IO_ERROR,
    UNAVAILABLE
};

struct PolicyExtent {
    std::string path;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t aligned_size = 0;
    int shard = 0;
};

struct PolicyRegionLease {
    std::string boot;
    std::string policy;
    std::string owner;
    std::string nonce;
    uint64_t id = 0;
    PolicyExtent extent;
    uint64_t object_size = 0;
    uint64_t stride = 0;
    uint64_t slots = 0;
    uint64_t committed = 0;
};

struct PolicyObjectLocation {
    bool found = false;
    uint64_t region_id = 0;
    uint64_t ordinal = 0;
};

struct PolicyRegionReply {
    PolicyRegionCode code = PolicyRegionCode::UNAVAILABLE;
    PolicyRegionLease region;
    uint64_t read_id = 0;
    std::vector<PolicyObjectLocation> objects;
    std::vector<PolicyRegionLease> regions;
};

struct PolicyRegionStats {
    uint64_t reserve_calls = 0, publish_calls = 0, acquire_calls = 0;
    uint64_t release_calls = 0, close_calls = 0, revoke_calls = 0;
    uint64_t reclaim_calls = 0, allocation_ops = 0;
    uint64_t publication_items = 0, lookup_items = 0;
    uint64_t live_regions = 0, live_objects = 0, pending_policies = 0;
    uint64_t retained_bytes = 0, reclaimed_bytes = 0;
    uint64_t join_calls = 0, leave_calls = 0, gc_failures = 0;
};

struct PolicyRegionConfig {
    bool enabled = false;
    std::string boot, fsdir, adapter;
    uint64_t shard_capacity = 0, alignment = 0;
    int shard_count = 0;
};

class PolicyRegionRegistry {
   public:
    using Allocate = std::function<PolicyExtent(const std::string&, uint64_t)>;
    using Free = std::function<void(const std::string&, const PolicyExtent&)>;
    PolicyRegionRegistry(std::string boot, Allocate allocate, Free free);
    PolicyRegionReply Reserve(const std::string& policy,
                              const std::string& owner,
                              const std::string& nonce, uint64_t object_size,
                              uint64_t stride, uint64_t slots);
    PolicyRegionReply Publish(const std::string& boot, uint64_t region,
                              const std::string& owner, uint64_t start,
                              const std::vector<std::string>& keys);
    PolicyRegionReply Acquire(const std::string& boot,
                              const std::string& policy,
                              const std::string& reader,
                              const std::vector<std::string>& keys);
    PolicyRegionCode Release(const std::string& boot, uint64_t read_id,
                             const std::string& reader);
    PolicyRegionCode Close(const std::string& boot, uint64_t region,
                           const std::string& owner);
    PolicyRegionCode Revoke(const std::string& boot, const std::string& policy);
    PolicyRegionCode Reclaim(const std::string& boot,
                             const std::string& policy);
    PolicyRegionCode Join(const std::string& boot, const std::string& policy,
                          const std::string& owner);
    PolicyRegionCode Leave(const std::string& boot, const std::string& policy,
                           const std::string& owner);
    // Called by the master's GC thread; frees outside the registry mutex.
    void Collect();
    PolicyRegionStats Stats() const;
    const std::string& Boot() const { return boot_; }

   private:
    struct Region {
        PolicyRegionLease lease;
        std::vector<std::string> keys;
        uint64_t readers = 0;
        bool writer_open = true;
    };
    struct Locator {
        uint64_t region = 0, ordinal = 0;
    };
    struct Policy {
        bool active = true;
        std::unordered_set<std::string> members;
        std::unordered_map<uint64_t, std::shared_ptr<Region>> regions;
        std::unordered_map<std::string, Locator> index;
    };
    struct Read {
        std::string owner;
        std::vector<std::shared_ptr<Region>> regions;
    };
    std::string boot_;
    Allocate allocate_;
    Free free_;
    mutable std::mutex mutex_;
    uint64_t next_region_ = 1, next_read_ = 1;
    std::unordered_map<std::string, std::shared_ptr<Policy>> policies_;
    // Tombstones fence replayed old policy identities after GC.
    std::unordered_set<std::string> revoked_;
    std::unordered_map<uint64_t, std::shared_ptr<Region>> regions_;
    std::unordered_map<uint64_t, Read> reads_;
    std::vector<std::shared_ptr<Policy>> garbage_;
    PolicyRegionStats stats_;
};
}  // namespace mooncake
