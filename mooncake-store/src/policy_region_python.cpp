#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>

#include "policy_region.h"
#include "master_client.h"
#include "client_service.h"
#include "common/client_buffer_allocation.h"
#include "device/accelerator_registry.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/posix_fs_adapter.h"
#ifdef USE_3FS
#include "storage/distributed/hf3fs_adapter.h"
#endif

namespace py = pybind11;
namespace mooncake {
namespace {
template <class T>
T Checked(tl::expected<T, ErrorCode> result) {
    if (!result)
        throw std::runtime_error(
            "native RPC error: " +
            std::to_string(static_cast<int>(result.error())));
    return std::move(*result);
}
void Check(PolicyRegionCode code) {
    if (code != PolicyRegionCode::OK)
        throw std::runtime_error("policy region error: " +
                                 std::to_string(static_cast<int>(code)));
}
DistributedFSDescriptor Descriptor(const PolicyRegionLease& r,
                                   uint64_t ordinal) {
    if (ordinal >= r.slots || r.object_size > r.stride ||
        ordinal > (UINT64_MAX - r.extent.offset) / r.stride)
        throw std::runtime_error("invalid region geometry");
    return {r.extent.path, r.extent.offset + ordinal * r.stride, r.object_size,
            r.stride, r.extent.shard};
}
std::vector<std::vector<Slice>> Slices(
    const std::vector<std::vector<uintptr_t>>& pointers,
    const std::vector<std::vector<uint64_t>>& sizes, size_t count) {
    if (pointers.size() != count || sizes.size() != count)
        throw std::invalid_argument("batch length mismatch");
    std::vector<std::vector<Slice>> result(count);
    for (size_t k = 0; k < count; k++) {
        if (pointers[k].empty() || pointers[k].size() != sizes[k].size())
            throw std::invalid_argument("slice length mismatch");
        for (size_t s = 0; s < pointers[k].size(); s++) {
            if (!pointers[k][s] || !sizes[k][s] ||
                sizes[k][s] > UINT64_MAX - pointers[k][s])
                throw std::invalid_argument("invalid slice");
            result[k].push_back(
                {reinterpret_cast<void*>(pointers[k][s]), sizes[k][s]});
        }
    }
    return result;
}
uint64_t Bytes(const std::vector<Slice>& slices) {
    uint64_t bytes = 0;
    for (const auto& s : slices) {
        if (s.size > UINT64_MAX - bytes)
            throw std::invalid_argument("slice size overflow");
        bytes += s.size;
    }
    return bytes;
}
}  // namespace

// Each operation holds the client mutex through native completion. Close/drain
// cannot acknowledge quiescence while this client's I/O still uses a region.
class NativePolicyClient {
   public:
    NativePolicyClient(const std::string& master, const std::string& hostname,
                       bool regions, uint64_t memory_bytes = 67108864)
        : uuid_(generate_uuid()), master_(uuid_), regions_enabled_(regions) {
        auto rc = master_.Connect(master);
        if (rc != ErrorCode::OK)
            throw std::runtime_error("Master connect failed");
        config_ = DistributedStorageConfig::FromEnvironment();
        if (regions ||
            (std::getenv("MOONCAKE_POLICY_REGIONS") &&
             std::string_view(std::getenv("MOONCAKE_POLICY_REGIONS")) == "1")) {
            info_ = Checked(master_.RegionInfo());
            if (!info_.enabled)
                throw std::runtime_error("Master policy regions unavailable");
            config_.fsdir = info_.fsdir;
            config_.fs_adapter_type = info_.adapter;
            config_.shard_capacity = info_.shard_capacity;
            config_.shard_count = info_.shard_count;
            config_.alignment = info_.alignment;
        }
        auto client = Client::Create(hostname, "P2PHANDSHAKE", "tcp",
                                     std::nullopt, master);
        if (!client)
            throw std::runtime_error("native Client initialization failed");
        client_ = *client;
        FileStorageConfig file;
        file.storage_backend_type = StorageBackendType::kDistributed;
        file.storage_filepath = config_.fsdir;
        std::unique_ptr<FileSystemAdapter> adapter;
        if (config_.fs_adapter_type == "posix")
            adapter = std::make_unique<PosixFsAdapter>();
#ifdef USE_3FS
        else if (config_.fs_adapter_type == "hf3fs")
            adapter = std::make_unique<Hf3fsAdapter>();
#endif
        else
            throw std::runtime_error("unsupported adapter");
        backend_ = std::make_shared<DistributedStorageBackend>(
            file, config_, std::move(adapter));
        auto init = backend_->Init();
        if (!init) throw std::runtime_error("DFS init failed");
        client_->SetDfsStorageBackend(backend_);
        if (memory_bytes) {
            segment_bytes_ = memory_bytes;
            segment_ = allocate_buffer_allocator_memory(memory_bytes);
            if (!segment_) throw std::runtime_error("memory allocation failed");
            if (!client_->MountSegment(segment_, memory_bytes, "tcp")) {
                free_memory("tcp", segment_);
                segment_ = nullptr;
                throw std::runtime_error("memory segment mount failed");
            }
        }
    }
    ~NativePolicyClient() {
        try {
            Close();
        } catch (...) {
        }
    }
    std::string Owner() const { return UuidToString(uuid_); }
    PolicyRegionConfig Info() const { return info_; }
    PolicyRegionStats Stats() {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        return Checked(master_.RegionStats());
    }
    void Register(uintptr_t address, uint64_t size) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        if (!address || !size)
            throw std::invalid_argument("invalid registration");
        auto rc = client_->RegisterLocalMemory(reinterpret_cast<void*>(address),
                                               size, "*", true, false);
        if (!rc) throw std::runtime_error("native registration failed");
    }
    PolicyRegionLease Reserve(const std::string& policy,
                              const std::string& nonce, uint64_t object_size,
                              uint64_t slots) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        if (!object_size || object_size > UINT64_MAX - config_.alignment)
            throw std::invalid_argument("invalid object size");
        auto stride =
            ((object_size + config_.alignment - 1) / config_.alignment) *
            config_.alignment;
        auto reply = Checked(master_.RegionReserve(policy, Owner(), nonce,
                                                   object_size, stride, slots));
        Check(reply.code);
        owned_.try_emplace(reply.region.id, reply.region);
        return reply.region;
    }
    std::vector<int64_t> Put(
        const PolicyRegionLease& lease, const std::vector<std::string>& keys,
        const std::vector<std::vector<uintptr_t>>& pointers,
        const std::vector<std::vector<uint64_t>>& sizes) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        auto found = owned_.find(lease.id);
        if (lease.boot != info_.boot || found == owned_.end() ||
            failed_.contains(lease.id))
            throw std::runtime_error("closed or uncertain writer");
        auto& r = found->second;
        auto slices = Slices(pointers, sizes, keys.size());
        if (keys.empty() || keys.size() > r.slots - r.committed)
            throw std::invalid_argument("region full");
        for (const auto& s : slices)
            if (Bytes(s) != r.object_size)
                throw std::invalid_argument("region object size mismatch");
        std::vector<DistributedFSDescriptor> descriptors;
        std::vector<const std::vector<Slice>*> addresses;
        for (size_t i = 0; i < keys.size(); i++) {
            descriptors.push_back(Descriptor(r, r.committed + i));
            addresses.push_back(&slices[i]);
        }
        auto written = client_->WritePolicyDfs(keys, descriptors, addresses);
        if (written.size() != keys.size()) {
            failed_.insert(r.id);
            throw std::runtime_error("DFS completion size mismatch");
        }
        for (auto rc : written)
            if (rc != ErrorCode::OK) {
                failed_.insert(r.id);
                throw std::runtime_error(
                    "DFS completion failed; publication suppressed");
            }
        if (fault_ == "completion") {
            fault_.clear();
            failed_.insert(r.id);
            throw std::runtime_error("injected uncertain DFS completion");
        }
        // The cursor advances only after an acknowledged publication. A lost
        // ACK fences this writer; retrying writes could overwrite published KV.
        try {
            auto reply = Checked(master_.RegionPublish(
                info_.boot, r.id, Owner(), r.committed, keys));
            Check(reply.code);
            if (fault_ == "publication_ack") {
                fault_.clear();
                throw std::runtime_error("injected lost publication ACK");
            }
            r = reply.region;
        } catch (...) {
            failed_.insert(r.id);
            throw;
        }
        return std::vector<int64_t>(keys.size(), 0);
    }
    PolicyRegionReply Acquire(const std::string& policy,
                              const std::vector<std::string>& keys) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        auto reply =
            Checked(master_.RegionAcquire(info_.boot, policy, Owner(), keys));
        Check(reply.code);
        if (reply.read_id) {
            views_.emplace(reply.read_id, reply);
            std::unordered_map<std::string, size_t> positions;
            positions.reserve(keys.size());
            for (size_t k = 0; k < keys.size(); ++k)
                positions.emplace(keys[k], k);
            view_keys_.emplace(reply.read_id, std::move(positions));
        }
        return reply;
    }
    std::vector<int64_t> Get(
        const PolicyRegionReply& view, const std::vector<std::string>& keys,
        const std::vector<std::vector<uintptr_t>>& pointers,
        const std::vector<std::vector<uint64_t>>& sizes) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        auto found = views_.find(view.read_id);
        if (!view.read_id || found == views_.end())
            throw std::runtime_error("unpinned view");
        const auto& pinned = found->second;
        auto slices = Slices(pointers, sizes, keys.size());
        std::vector<DfsReadRequest> requests;
        std::vector<std::vector<char>> host(keys.size());
        std::vector<bool> staged(keys.size(), false);
        auto accelerator =
            device::GetAcceleratorRegistry().RuntimeAccelerators();
        std::vector<DistributedFSDescriptor> descriptors;
        for (size_t k = 0; k < keys.size(); k++) {
            const auto& original = view_keys_.at(view.read_id);
            auto position = original.find(keys[k]);
            if (position == original.end())
                throw std::runtime_error("key not bound to view");
            const auto& location = pinned.objects[position->second];
            auto geometry = std::find_if(
                pinned.regions.begin(), pinned.regions.end(),
                [&](const auto& r) { return r.id == location.region_id; });
            if (!location.found || geometry == pinned.regions.end() ||
                Bytes(slices[k]) != geometry->object_size)
                throw std::runtime_error("missing or invalid object");
            auto descriptor = Descriptor(*geometry, location.ordinal);
            for (const auto& slice : slices[k])
                if (accelerator.FindDeviceForPointer(slice.ptr))
                    staged[k] = true;
            if (staged[k]) {
                host[k].resize(descriptor.object_size);
                requests.push_back(
                    {keys[k], descriptor, {{host[k].data(), host[k].size()}}});
            } else
                requests.push_back({keys[k], descriptor, slices[k]});
        }
        auto results = backend_->BatchRead(requests);
        if (results.size() != keys.size())
            throw std::runtime_error("DFS read completion size mismatch");
        std::vector<int64_t> bytes;
        for (size_t k = 0; k < keys.size(); k++) {
            if (!results[k]) throw std::runtime_error("DFS read failed");
            if (!staged[k]) {
                bytes.push_back(Bytes(slices[k]));
                continue;
            }
            size_t offset = 0;
            for (const auto& slice : slices[k]) {
                device::PointerInfo pointer{};
                auto* device =
                    accelerator.FindDeviceForPointer(slice.ptr, &pointer);
                if (device) {
                    device->SetContext(pointer.device_id);
                    if (!device->Copy(slice.ptr, host[k].data() + offset,
                                      slice.size,
                                      device::CopyDirection::kHostToDevice))
                        throw std::runtime_error("H2D restore failed");
                } else
                    std::memcpy(slice.ptr, host[k].data() + offset, slice.size);
                offset += slice.size;
            }
            bytes.push_back(host[k].size());
        }
        return bytes;
    }
    void Release(uint64_t read_id) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        Check(Checked(master_.RegionRelease(info_.boot, read_id, Owner())));
        views_.erase(read_id);
        view_keys_.erase(read_id);
    }
    void CloseRegion(uint64_t id) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        Check(Checked(master_.RegionClose(info_.boot, id, Owner())));
        owned_.erase(id);
        failed_.erase(id);
    }
    void Join(const std::string& policy) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        Check(Checked(master_.RegionJoin(info_.boot, policy, Owner())));
        joined_.insert(policy);
    }
    void Leave(const std::string& policy) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        Check(Checked(master_.RegionLeave(info_.boot, policy, Owner())));
        joined_.erase(policy);
    }
    void Revoke(const std::string& policy) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        Check(Checked(master_.RegionRevoke(info_.boot, policy)));
    }
    PolicyRegionCode Reclaim(const std::string& policy) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        return Checked(master_.RegionReclaim(info_.boot, policy));
    }
    // Explicit deterministic fault points belong only to the optional PoC SDK.
    void InjectFault(const std::string& fault) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        if (fault != "completion" && fault != "publication_ack")
            throw std::invalid_argument("unknown fault point");
        fault_ = fault;
    }
    void Flush() {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        FlushLocked();
    }
    void Drain() {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        DrainLocked();
    }
    void Close() {
        std::lock_guard lock(mutex_);
        if (closed_) return;
        if (regions_enabled_) DrainLocked();
        if (segment_) {
            (void)client_->UnmountSegment(segment_, segment_bytes_);
            free_memory("tcp", segment_);
            segment_ = nullptr;
        }
        client_.reset();
        backend_.reset();
        closed_ = true;
    }
    std::vector<int64_t> ObjectPut(
        const std::vector<std::string>& keys,
        const std::vector<std::vector<uintptr_t>>& pointers,
        const std::vector<std::vector<uint64_t>>& sizes) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        auto slices = Slices(pointers, sizes, keys.size());
        ReplicateConfig config;
        config.replica_num = 1;
        config.dfs_replica_num = 1;
        auto results = client_->BatchPut(keys, slices, config);
        std::vector<int64_t> ret;
        for (const auto& result : results) {
            if (!result) throw std::runtime_error("object PUT failed");
            ret.push_back(0);
        }
        return ret;
    }
    std::vector<int> ObjectExists(const std::vector<std::string>& keys) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        auto results = client_->BatchIsExist(keys);
        std::vector<int> ret;
        for (const auto& r : results) {
            if (!r) throw std::runtime_error("object lookup failed");
            ret.push_back(*r);
        }
        return ret;
    }
    // Force DFS for a comparable storage restore. This does not benchmark the
    // ordinary memory-hit selection of the baseline's extra MEMORY replica.
    std::vector<int64_t> ObjectGet(
        const std::vector<std::string>& keys,
        const std::vector<std::vector<uintptr_t>>& pointers,
        const std::vector<std::vector<uint64_t>>& sizes) {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        auto slices = Slices(pointers, sizes, keys.size());
        auto queries = client_->BatchQuery(keys);
        if (queries.size() != keys.size())
            throw std::runtime_error("object query size mismatch");
        std::vector<DfsReadRequest> requests;
        for (size_t k = 0; k < keys.size(); k++) {
            if (!queries[k]) throw std::runtime_error("object query failed");
            bool found = false;
            for (const auto& r : queries[k]->replicas)
                if (r.is_dfs_replica()) {
                    requests.push_back(
                        {keys[k], r.get_dfs_descriptor(), slices[k]});
                    found = true;
                    break;
                }
            if (!found) throw std::runtime_error("no DFS replica");
        }
        auto reads = backend_->BatchRead(requests);
        if (reads.size() != keys.size())
            throw std::runtime_error("object read completion size mismatch");
        std::vector<int64_t> ret;
        for (size_t k = 0; k < reads.size(); k++) {
            if (!reads[k]) throw std::runtime_error("object read failed");
            ret.push_back(Bytes(slices[k]));
        }
        return ret;
    }
    int64_t ObjectReset() {
        std::lock_guard lock(mutex_);
        EnsureOpen();
        return Checked(client_->RemoveAll(true));
    }

   private:
    void EnsureOpen() const {
        if (closed_) throw std::runtime_error("closed native client");
    }
    void FlushLocked() {
        for (auto it = owned_.begin(); it != owned_.end();) {
            Check(Checked(master_.RegionClose(info_.boot, it->first, Owner())));
            it = owned_.erase(it);
        }
        for (auto it = views_.begin(); it != views_.end();) {
            Check(
                Checked(master_.RegionRelease(info_.boot, it->first, Owner())));
            view_keys_.erase(it->first);
            it = views_.erase(it);
        }
        failed_.clear();
    }
    void DrainLocked() {
        FlushLocked();
        for (auto it = joined_.begin(); it != joined_.end();) {
            Check(Checked(master_.RegionLeave(info_.boot, *it, Owner())));
            it = joined_.erase(it);
        }
    }
    std::string fault_;
    std::unordered_set<std::string> joined_;
    UUID uuid_;
    MasterClient master_;
    bool regions_enabled_ = false, closed_ = false;
    PolicyRegionConfig info_;
    DistributedStorageConfig config_;
    std::shared_ptr<Client> client_;
    std::shared_ptr<DistributedStorageBackend> backend_;
    void* segment_ = nullptr;
    uint64_t segment_bytes_ = 0;
    std::mutex mutex_;
    std::unordered_map<uint64_t, PolicyRegionLease> owned_;
    std::unordered_map<uint64_t, PolicyRegionReply> views_;
    std::unordered_map<uint64_t, std::unordered_map<std::string, size_t>>
        view_keys_;
    std::unordered_set<uint64_t> failed_;
};
}  // namespace mooncake

PYBIND11_MODULE(policy_region_native, m) {
    using namespace mooncake;
    py::enum_<PolicyRegionCode>(m, "Code")
        .value("OK", PolicyRegionCode::OK)
        .value("STALE", PolicyRegionCode::STALE)
        .value("BUSY", PolicyRegionCode::BUSY)
        .value("INVALID", PolicyRegionCode::INVALID);
    py::class_<PolicyExtent>(m, "Extent")
        .def_readonly("path", &PolicyExtent::path)
        .def_readonly("offset", &PolicyExtent::offset)
        .def_readonly("size", &PolicyExtent::size);
    py::class_<PolicyRegionLease>(m, "Region")
        .def_readonly("boot", &PolicyRegionLease::boot)
        .def_readonly("id", &PolicyRegionLease::id)
        .def_readonly("policy", &PolicyRegionLease::policy)
        .def_readonly("extent", &PolicyRegionLease::extent)
        .def_readonly("object_size", &PolicyRegionLease::object_size)
        .def_readonly("stride", &PolicyRegionLease::stride)
        .def_readonly("slots", &PolicyRegionLease::slots)
        .def_readonly("committed", &PolicyRegionLease::committed);
    py::class_<PolicyObjectLocation>(m, "Location")
        .def_readonly("found", &PolicyObjectLocation::found)
        .def_readonly("region_id", &PolicyObjectLocation::region_id)
        .def_readonly("ordinal", &PolicyObjectLocation::ordinal);
    py::class_<PolicyRegionReply>(m, "View")
        .def_readonly("read_id", &PolicyRegionReply::read_id)
        .def_readonly("objects", &PolicyRegionReply::objects)
        .def_readonly("regions", &PolicyRegionReply::regions);
    py::class_<PolicyRegionConfig>(m, "Config")
        .def_readonly("boot", &PolicyRegionConfig::boot)
        .def_readonly("fsdir", &PolicyRegionConfig::fsdir);
    py::class_<PolicyRegionStats>(m, "Stats")
#define FIELD(x) .def_readonly(#x, &PolicyRegionStats::x)
        FIELD(reserve_calls) FIELD(publish_calls) FIELD(acquire_calls)
            FIELD(release_calls) FIELD(close_calls) FIELD(revoke_calls)
                FIELD(reclaim_calls) FIELD(allocation_ops)
                    FIELD(publication_items) FIELD(lookup_items)
                        FIELD(live_regions) FIELD(live_objects)
                            FIELD(pending_policies) FIELD(retained_bytes)
                                FIELD(reclaimed_bytes) FIELD(join_calls)
                                    FIELD(leave_calls) FIELD(gc_failures);
#undef FIELD
    py::class_<NativePolicyClient>(m, "Client")
        .def(py::init<const std::string&, const std::string&, bool, uint64_t>(),
             py::arg("master"), py::arg("hostname"), py::arg("regions"),
             py::arg("memory_bytes") = 67108864)
        .def("owner", &NativePolicyClient::Owner)
        .def("info", &NativePolicyClient::Info)
        .def("stats", &NativePolicyClient::Stats,
             py::call_guard<py::gil_scoped_release>())
        .def("register_buffer", &NativePolicyClient::Register,
             py::call_guard<py::gil_scoped_release>())
        .def("reserve", &NativePolicyClient::Reserve,
             py::call_guard<py::gil_scoped_release>())
        .def("put", &NativePolicyClient::Put,
             py::call_guard<py::gil_scoped_release>())
        .def("acquire", &NativePolicyClient::Acquire,
             py::call_guard<py::gil_scoped_release>())
        .def("get", &NativePolicyClient::Get,
             py::call_guard<py::gil_scoped_release>())
        .def("release", &NativePolicyClient::Release,
             py::call_guard<py::gil_scoped_release>())
        .def("close_region", &NativePolicyClient::CloseRegion,
             py::call_guard<py::gil_scoped_release>())
        .def("join", &NativePolicyClient::Join,
             py::call_guard<py::gil_scoped_release>())
        .def("leave", &NativePolicyClient::Leave,
             py::call_guard<py::gil_scoped_release>())
        .def("revoke", &NativePolicyClient::Revoke,
             py::call_guard<py::gil_scoped_release>())
        .def("reclaim", &NativePolicyClient::Reclaim,
             py::call_guard<py::gil_scoped_release>())
        .def("inject_fault", &NativePolicyClient::InjectFault,
             py::call_guard<py::gil_scoped_release>())
        .def("flush", &NativePolicyClient::Flush,
             py::call_guard<py::gil_scoped_release>())
        .def("drain", &NativePolicyClient::Drain,
             py::call_guard<py::gil_scoped_release>())
        .def("close", &NativePolicyClient::Close,
             py::call_guard<py::gil_scoped_release>())
        .def("object_put", &NativePolicyClient::ObjectPut,
             py::call_guard<py::gil_scoped_release>())
        .def("object_exists", &NativePolicyClient::ObjectExists,
             py::call_guard<py::gil_scoped_release>())
        .def("object_get", &NativePolicyClient::ObjectGet,
             py::call_guard<py::gil_scoped_release>())
        .def("object_reset", &NativePolicyClient::ObjectReset,
             py::call_guard<py::gil_scoped_release>());
}
