#include "policy_region.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mooncake {
namespace {
PolicyRegionReply Reply(PolicyRegionCode code) {
    PolicyRegionReply reply;
    reply.code = code;
    return reply;
}
std::string AllocationKey(const PolicyRegionLease& lease) {
    return "__policy_region__" + lease.boot + ":" + std::to_string(lease.id);
}
}  // namespace

PolicyRegionRegistry::PolicyRegionRegistry(std::string boot, Allocate allocate,
                                           Free free)
    : boot_(std::move(boot)),
      allocate_(std::move(allocate)),
      free_(std::move(free)) {
    if (boot_.empty() || !allocate_ || !free_)
        throw std::invalid_argument("invalid region registry");
}

PolicyRegionReply PolicyRegionRegistry::Reserve(const std::string& policy,
                                                const std::string& owner,
                                                const std::string& nonce,
                                                uint64_t object_size,
                                                uint64_t stride,
                                                uint64_t slots) {
    std::lock_guard lock(mutex_);
    ++stats_.reserve_calls;
    if (policy.empty() || policy.size() > 1024 || owner.empty() ||
        owner.size() > 256 || nonce.empty() || nonce.size() > 256 ||
        !object_size || stride < object_size || !slots || slots > (1U << 20) ||
        slots > UINT64_MAX / stride || next_region_ == UINT64_MAX)
        return Reply(PolicyRegionCode::INVALID);
    if (revoked_.contains(policy)) return Reply(PolicyRegionCode::STALE);
    auto current = policies_.find(policy);
    if (current != policies_.end()) {
        for (const auto& [id, region] : current->second->regions) {
            const auto& lease = region->lease;
            if (lease.owner != owner || lease.nonce != nonce) continue;
            if (!region->writer_open) return Reply(PolicyRegionCode::STALE);
            if (lease.object_size != object_size || lease.stride != stride ||
                lease.slots != slots)
                return Reply(PolicyRegionCode::INVALID);
            auto result = Reply(PolicyRegionCode::OK);
            result.region = lease;
            return result;
        }
    }
    auto region = std::make_shared<Region>();
    region->lease = {boot_, policy,      owner,  nonce, next_region_++,
                     {},    object_size, stride, slots, 0};
    try {
        region->lease.extent =
            allocate_(AllocationKey(region->lease), stride * slots);
    } catch (...) {
        return Reply(PolicyRegionCode::IO_ERROR);
    }
    const auto& extent = region->lease.extent;
    if (extent.path.empty() || extent.size != stride * slots ||
        extent.aligned_size < extent.size ||
        extent.offset > UINT64_MAX - extent.aligned_size) {
        // A malformed allocator result cannot be used. Do not recycle an
        // uncertain range based on a descriptor that failed validation.
        return Reply(PolicyRegionCode::IO_ERROR);
    }
    auto& epoch = policies_[policy];
    if (!epoch) epoch = std::make_shared<Policy>();
    epoch->regions.emplace(region->lease.id, region);
    regions_.emplace(region->lease.id, region);
    ++stats_.allocation_ops;
    ++stats_.live_regions;
    stats_.retained_bytes += extent.aligned_size;
    auto result = Reply(PolicyRegionCode::OK);
    result.region = region->lease;
    return result;
}

PolicyRegionReply PolicyRegionRegistry::Publish(
    const std::string& boot, uint64_t id, const std::string& owner,
    uint64_t start, const std::vector<std::string>& keys) {
    std::lock_guard lock(mutex_);
    ++stats_.publish_calls;
    if (boot != boot_) return Reply(PolicyRegionCode::STALE);
    auto found = regions_.find(id);
    if (found == regions_.end()) return Reply(PolicyRegionCode::NOT_FOUND);
    auto& region = *found->second;
    if (region.lease.owner != owner || !region.writer_open ||
        revoked_.contains(region.lease.policy))
        return Reply(PolicyRegionCode::STALE);
    if (keys.empty() || start > region.lease.slots ||
        keys.size() > region.lease.slots - start)
        return Reply(PolicyRegionCode::INVALID);
    auto& policy = *policies_.at(region.lease.policy);
    if (start < region.lease.committed) {
        if (keys.size() > region.lease.committed - start)
            return Reply(PolicyRegionCode::INVALID);
        for (size_t k = 0; k < keys.size(); ++k)
            if (region.keys[start + k] != keys[k])
                return Reply(PolicyRegionCode::INVALID);
        auto result = Reply(PolicyRegionCode::OK);
        result.region = region.lease;
        return result;
    }
    if (start != region.lease.committed)
        return Reply(PolicyRegionCode::INVALID);
    std::unordered_set<std::string> unique;
    for (const auto& key : keys) {
        if (key.empty() || key.size() > 4096 || policy.index.contains(key) ||
            !unique.insert(key).second)
            return Reply(PolicyRegionCode::INVALID);
    }
    // Validate the whole batch before modifying the public index. Readers see
    // either the old frontier or the complete new frontier under the same lock.
    size_t inserted = 0;
    try {
        auto staged = keys;
        region.keys.reserve(region.keys.size() + keys.size());
        policy.index.reserve(policy.index.size() + keys.size());
        for (size_t k = 0; k < keys.size(); ++k) {
            policy.index.emplace(keys[k], Locator{id, start + k});
            ++inserted;
        }
        for (auto& key : staged) region.keys.push_back(std::move(key));
    } catch (...) {
        for (size_t k = 0; k < inserted; ++k) policy.index.erase(keys[k]);
        return Reply(PolicyRegionCode::IO_ERROR);
    }
    region.lease.committed += keys.size();
    stats_.publication_items += keys.size();
    stats_.live_objects += keys.size();
    auto result = Reply(PolicyRegionCode::OK);
    result.region = region.lease;
    return result;
}

PolicyRegionReply PolicyRegionRegistry::Acquire(
    const std::string& boot, const std::string& policy,
    const std::string& reader, const std::vector<std::string>& keys) {
    std::lock_guard lock(mutex_);
    ++stats_.acquire_calls;
    if (boot != boot_ || revoked_.contains(policy))
        return Reply(PolicyRegionCode::STALE);
    if (reader.empty() || reader.size() > 256 || keys.size() > (1U << 20))
        return Reply(PolicyRegionCode::INVALID);
    auto result = Reply(PolicyRegionCode::OK);
    result.objects.resize(keys.size());
    stats_.lookup_items += keys.size();
    auto found = policies_.find(policy);
    if (found == policies_.end()) return result;
    Read read;
    read.owner = reader;
    std::unordered_set<uint64_t> pinned;
    for (size_t k = 0; k < keys.size(); ++k) {
        auto location = found->second->index.find(keys[k]);
        if (location == found->second->index.end()) continue;
        auto region = found->second->regions.at(location->second.region);
        if (location->second.ordinal >= region->lease.committed) continue;
        result.objects[k] = {true, region->lease.id, location->second.ordinal};
        if (pinned.insert(region->lease.id).second) {
            read.regions.push_back(region);
            result.regions.push_back(region->lease);
        }
    }
    if (!read.regions.empty()) {
        if (next_read_ == UINT64_MAX)
            return Reply(PolicyRegionCode::UNAVAILABLE);
        for (const auto& region : read.regions) ++region->readers;
        result.read_id = next_read_++;
        reads_.emplace(result.read_id, std::move(read));
    }
    return result;
}

PolicyRegionCode PolicyRegionRegistry::Release(const std::string& boot,
                                               uint64_t id,
                                               const std::string& reader) {
    std::lock_guard lock(mutex_);
    ++stats_.release_calls;
    if (boot != boot_) return PolicyRegionCode::STALE;
    auto found = reads_.find(id);
    if (found == reads_.end())
        return PolicyRegionCode::OK;  // Retry after acknowledged release.
    if (found->second.owner != reader) return PolicyRegionCode::STALE;
    for (const auto& region : found->second.regions) --region->readers;
    reads_.erase(found);
    return PolicyRegionCode::OK;
}

PolicyRegionCode PolicyRegionRegistry::Close(const std::string& boot,
                                             uint64_t id,
                                             const std::string& owner) {
    std::lock_guard lock(mutex_);
    ++stats_.close_calls;
    if (boot != boot_) return PolicyRegionCode::STALE;
    auto found = regions_.find(id);
    if (found == regions_.end()) return PolicyRegionCode::NOT_FOUND;
    if (found->second->lease.owner != owner) return PolicyRegionCode::STALE;
    found->second->writer_open = false;
    return PolicyRegionCode::OK;
}

PolicyRegionCode PolicyRegionRegistry::Revoke(const std::string& boot,
                                              const std::string& policy) {
    std::lock_guard lock(mutex_);
    ++stats_.revoke_calls;
    if (boot != boot_) return PolicyRegionCode::STALE;
    if (policy.empty()) return PolicyRegionCode::INVALID;
    revoked_.insert(policy);
    auto found = policies_.find(policy);
    if (found != policies_.end()) found->second->active = false;
    return PolicyRegionCode::OK;
}

PolicyRegionCode PolicyRegionRegistry::Reclaim(const std::string& boot,
                                               const std::string& policy) {
    std::lock_guard lock(mutex_);
    ++stats_.reclaim_calls;
    if (boot != boot_) return PolicyRegionCode::STALE;
    if (!revoked_.contains(policy)) return PolicyRegionCode::BUSY;
    auto found = policies_.find(policy);
    if (found == policies_.end()) return PolicyRegionCode::OK;
    for (const auto& [id, region] : found->second->regions)
        if (region->writer_open || region->readers)
            return PolicyRegionCode::BUSY;
    // Only O(regions) work in this RPC. The per-key index is destroyed by
    // Collect.
    for (const auto& [id, region] : found->second->regions) regions_.erase(id);
    garbage_.push_back(std::move(found->second));
    policies_.erase(found);
    ++stats_.pending_policies;
    return PolicyRegionCode::OK;
}

PolicyRegionCode PolicyRegionRegistry::Join(const std::string& boot,
                                            const std::string& policy,
                                            const std::string& owner) {
    std::lock_guard lock(mutex_);
    ++stats_.join_calls;
    if (boot != boot_ || revoked_.contains(policy))
        return PolicyRegionCode::STALE;
    if (policy.empty() || policy.size() > 1024 || owner.empty() ||
        owner.size() > 256)
        return PolicyRegionCode::INVALID;
    auto& epoch = policies_[policy];
    if (!epoch) epoch = std::make_shared<Policy>();
    epoch->members.insert(owner);
    return PolicyRegionCode::OK;
}
PolicyRegionCode PolicyRegionRegistry::Leave(const std::string& boot,
                                             const std::string& policy,
                                             const std::string& owner) {
    std::unique_lock lock(mutex_);
    ++stats_.leave_calls;
    if (boot != boot_) return PolicyRegionCode::STALE;
    auto found = policies_.find(policy);
    if (found == policies_.end() || !found->second->members.contains(owner))
        return PolicyRegionCode::OK;
    for (const auto& [id, region] : found->second->regions)
        if (region->lease.owner == owner && region->writer_open)
            return PolicyRegionCode::BUSY;
    for (const auto& [id, read] : reads_)
        if (read.owner == owner)
            for (const auto& region : read.regions)
                if (region->lease.policy == policy)
                    return PolicyRegionCode::BUSY;
    found->second->members.erase(owner);
    if (!found->second->members.empty()) return PolicyRegionCode::OK;
    revoked_.insert(policy);
    found->second->active = false;
    lock.unlock();
    // Explicit reader pins from non-members can still retain this generation.
    // A BUSY result here is safe retention, not permission to recycle storage.
    (void)Reclaim(boot, policy);
    return PolicyRegionCode::OK;
}

void PolicyRegionRegistry::Collect() {
    std::vector<std::shared_ptr<Policy>> garbage;
    {
        std::lock_guard lock(mutex_);
        garbage.swap(garbage_);
    }
    for (const auto& policy : garbage) {
        for (auto it = policy->regions.begin(); it != policy->regions.end();) {
            const auto region = it->second;
            try {
                free_(AllocationKey(region->lease), region->lease.extent);
            } catch (...) {
                std::lock_guard lock(mutex_);
                ++stats_.gc_failures;
                ++it;
                continue;
            }
            {
                std::lock_guard lock(mutex_);
                --stats_.live_regions;
                stats_.retained_bytes -= region->lease.extent.aligned_size;
                stats_.reclaimed_bytes += region->lease.extent.aligned_size;
            }
            it = policy->regions.erase(it);
        }
        std::lock_guard lock(mutex_);
        if (!policy->regions.empty()) {
            garbage_.push_back(policy);
            continue;
        }
        stats_.live_objects -= policy->index.size();
        --stats_.pending_policies;
    }
    // Index/key destruction takes place outside the registry mutex.
}

PolicyRegionStats PolicyRegionRegistry::Stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}
}  // namespace mooncake
