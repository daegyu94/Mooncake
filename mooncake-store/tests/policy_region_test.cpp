#include "policy_region.h"
#include <atomic>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace mooncake;
#define CHECK(x)                                \
    do {                                        \
        if (!(x)) throw std::runtime_error(#x); \
    } while (false)
#define OK(x) CHECK((x) == PolicyRegionCode::OK)

struct Fixture {
    uint64_t allocated = 0, freed = 0;
    bool fail = false;
    PolicyRegionRegistry registry{
        "boot-A",
        [this](const std::string&, uint64_t bytes) {
            if (fail) throw std::runtime_error("injected allocation failure");
            return PolicyExtent{"/dfs/shard", ++allocated * 65536, bytes, bytes,
                                0};
        },
        [this](const std::string&, const PolicyExtent&) {
            // Regression: reclamation must not hold the registry mutex.
            (void)registry.Stats();
            ++freed;
        }};
    PolicyRegionReply reserve(std::string owner = "writer",
                              std::string nonce = "n") {
        return registry.Reserve("weights-A", owner, nonce, 4096, 4096, 8);
    }
};

void TestGeometry() {
    Fixture f;
    CHECK(f.registry.Reserve("", "w", "n", 1, 1, 1).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Reserve("p", "", "n", 1, 1, 1).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Reserve("p", "w", "", 1, 1, 1).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Reserve("p", "w", "n", 0, 1, 1).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Reserve("p", "w", "n", 2, 1, 1).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Reserve("p", "w", "n", 1, 1, 0).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Reserve("p", "w", "n", 1, 2, UINT64_MAX).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.allocated == 0);
}
void TestReservationRetry() {
    Fixture f;
    auto a = f.reserve();
    OK(a.code);
    auto retry = f.reserve();
    OK(retry.code);
    CHECK(a.region.id == retry.region.id && f.allocated == 1);
    CHECK(f.registry.Reserve("weights-A", "writer", "n", 8192, 8192, 8).code ==
          PolicyRegionCode::INVALID);
    auto b = f.reserve("other");
    OK(b.code);
    CHECK(a.region.id != b.region.id);
}
void TestPublicationAndRead() {
    Fixture f;
    auto r = f.reserve();
    OK(r.code);
    const auto id = r.region.id;
    auto missing = f.registry.Acquire("boot-A", "weights-A", "reader", {"a"});
    OK(missing.code);
    CHECK(!missing.objects[0].found && missing.read_id == 0);
    OK(f.registry.Publish("boot-A", id, "writer", 0, {"a", "b"}).code);
    OK(f.registry.Publish("boot-A", id, "writer", 0, {"a", "b"}).code);
    CHECK(f.registry.Stats().publication_items == 2);
    CHECK(f.registry.Publish("boot-A", id, "writer", 0, {"wrong"}).code ==
          PolicyRegionCode::INVALID);
    CHECK(f.registry.Publish("boot-A", id, "writer", 3, {"hole"}).code ==
          PolicyRegionCode::INVALID);
    auto view = f.registry.Acquire("boot-A", "weights-A", "reader",
                                   {"a", "missing", "b", "a"});
    OK(view.code);
    CHECK(view.read_id != 0 && view.objects.size() == 4);
    CHECK(view.objects[0].ordinal == 0 && view.objects[2].ordinal == 1);
    CHECK(!view.objects[1].found && view.regions.size() == 1 &&
          view.regions[0].committed == 2);
    CHECK(f.registry.Release("boot-A", view.read_id, "wrong") ==
          PolicyRegionCode::STALE);
    OK(f.registry.Release("boot-A", view.read_id, "reader"));
    OK(f.registry.Release("boot-A", view.read_id, "reader"));
}
void TestAtomicDuplicateRejection() {
    Fixture f;
    auto a = f.reserve();
    auto b = f.reserve("other");
    OK(a.code);
    OK(b.code);
    OK(f.registry.Publish("boot-A", a.region.id, "writer", 0, {"a"}).code);
    CHECK(f.registry.Publish("boot-A", b.region.id, "other", 0, {"b", "a"})
              .code == PolicyRegionCode::INVALID);
    CHECK(f.registry.Publish("boot-A", b.region.id, "other", 0, {"c", "c"})
              .code == PolicyRegionCode::INVALID);
    auto v = f.registry.Acquire("boot-A", "weights-A", "r", {"b", "c"});
    OK(v.code);
    CHECK(!v.objects[0].found && !v.objects[1].found && v.read_id == 0);
    CHECK(f.registry.Stats().live_objects == 1);
}
void TestFenceAndDrain() {
    Fixture f;
    auto a = f.reserve();
    OK(a.code);
    CHECK(
        f.registry.Publish("boot-A", a.region.id, "impostor", 0, {"a"}).code ==
        PolicyRegionCode::STALE);
    OK(f.registry.Publish("boot-A", a.region.id, "writer", 0, {"a"}).code);
    auto v = f.registry.Acquire("boot-A", "weights-A", "r", {"a"});
    OK(v.code);
    CHECK(f.registry.Reclaim("boot-A", "weights-A") == PolicyRegionCode::BUSY);
    OK(f.registry.Revoke("boot-A", "weights-A"));
    OK(f.registry.Revoke("boot-A", "weights-A"));
    CHECK(f.reserve().code == PolicyRegionCode::STALE);
    CHECK(
        f.registry.Publish("boot-A", a.region.id, "writer", 1, {"late"}).code ==
        PolicyRegionCode::STALE);
    CHECK(f.registry.Acquire("boot-A", "weights-A", "new", {"a"}).code ==
          PolicyRegionCode::STALE);
    CHECK(v.objects[0]
              .found);  // An already pinned old-policy view remains valid.
    CHECK(f.registry.Reclaim("boot-A", "weights-A") == PolicyRegionCode::BUSY);
    CHECK(f.registry.Close("boot-A", a.region.id, "wrong") ==
          PolicyRegionCode::STALE);
    OK(f.registry.Close("boot-A", a.region.id, "writer"));
    CHECK(f.registry.Reclaim("boot-A", "weights-A") == PolicyRegionCode::BUSY);
    OK(f.registry.Release("boot-A", v.read_id, "r"));
    OK(f.registry.Reclaim("boot-A", "weights-A"));
    CHECK(f.freed == 0);
    CHECK(f.registry.Stats().pending_policies == 1);
    f.registry.Collect();
    CHECK(f.freed == 1);
    CHECK(f.registry.Stats().retained_bytes == 0);
    CHECK(f.registry.Stats().reclaimed_bytes == 32768);
    CHECK(f.reserve().code == PolicyRegionCode::STALE);
    OK(f.registry.Reclaim("boot-A", "weights-A"));
}
void TestBootAndClosedWriter() {
    Fixture f;
    auto a = f.reserve();
    OK(a.code);
    CHECK(
        f.registry.Publish("boot-old", a.region.id, "writer", 0, {"a"}).code ==
        PolicyRegionCode::STALE);
    CHECK(f.registry.Release("boot-old", 1, "r") == PolicyRegionCode::STALE);
    CHECK(f.registry.Close("boot-old", a.region.id, "writer") ==
          PolicyRegionCode::STALE);
    CHECK(f.registry.Revoke("boot-old", "weights-A") ==
          PolicyRegionCode::STALE);
    CHECK(f.registry.Reclaim("boot-old", "weights-A") ==
          PolicyRegionCode::STALE);
    OK(f.registry.Close("boot-A", a.region.id, "writer"));
    OK(f.registry.Close("boot-A", a.region.id, "writer"));
    CHECK(f.registry.Publish("boot-A", a.region.id, "writer", 0, {"a"}).code ==
          PolicyRegionCode::STALE);
    CHECK(f.reserve().code == PolicyRegionCode::STALE);
}
void TestOverlapAndLongRun() {
    Fixture f;
    auto a = f.reserve();
    OK(a.code);
    OK(f.registry.Publish("boot-A", a.region.id, "writer", 0, {"same"}).code);
    auto b = f.registry.Reserve("weights-B", "other", "n", 4096, 4096, 8);
    OK(b.code);
    OK(f.registry.Publish("boot-A", b.region.id, "other", 0, {"same"}).code);
    auto v = f.registry.Acquire("boot-A", "weights-B", "r", {"same"});
    OK(v.code);
    CHECK(v.regions[0].policy == "weights-B");
    OK(f.registry.Release("boot-A", v.read_id, "r"));
    for (int i = 0; i < 128; i++) {
        auto p = "long-" + std::to_string(i);
        auto r = f.registry.Reserve(p, "w", "n", 4096, 4096, 8);
        OK(r.code);
        OK(f.registry.Publish("boot-A", r.region.id, "w", 0, {"a", "b"}).code);
        OK(f.registry.Close("boot-A", r.region.id, "w"));
        OK(f.registry.Revoke("boot-A", p));
        OK(f.registry.Reclaim("boot-A", p));
        f.registry.Collect();
    }
    CHECK(f.registry.Stats().live_regions == 2 &&
          f.registry.Stats().live_objects == 2);
}
void TestAllocationFailureAndLimits() {
    Fixture f;
    f.fail = true;
    CHECK(f.reserve().code == PolicyRegionCode::IO_ERROR);
    CHECK(f.registry.Stats().live_regions == 0);
    f.fail = false;
    auto a = f.reserve();
    OK(a.code);
    CHECK(f.registry.Publish("boot-A", a.region.id, "writer", 0, {""}).code ==
          PolicyRegionCode::INVALID);
    std::vector<std::string> too_many;
    for (int i = 0; i < 9; i++) too_many.push_back(std::to_string(i));
    CHECK(
        f.registry.Publish("boot-A", a.region.id, "writer", 0, too_many).code ==
        PolicyRegionCode::INVALID);
    CHECK(f.registry.Stats().live_objects == 0);
}
void TestConcurrentOwners() {
    Fixture f;
    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    for (int w = 0; w < 4; w++)
        workers.emplace_back([&, w] {
            try {
                auto owner = "w" + std::to_string(w);
                auto r = f.reserve(owner);
                OK(r.code);
                for (int k = 0; k < 8; k++)
                    OK(f.registry
                           .Publish("boot-A", r.region.id, owner, k,
                                    {owner + "-" + std::to_string(k)})
                           .code);
                OK(f.registry.Close("boot-A", r.region.id, owner));
            } catch (...) {
                ++failures;
            }
        });
    for (auto& w : workers) w.join();
    CHECK(failures == 0);
    CHECK(f.registry.Stats().live_objects == 32 && f.allocated == 4);
}

void TestParticipation() {
    Fixture f;
    for (int w = 0; w < 4; w++)
        OK(f.registry.Join("boot-A", "weights-A", "w" + std::to_string(w)));
    OK(f.registry.Join("boot-A", "weights-A", "w0"));
    auto r = f.reserve("w0");
    OK(r.code);
    OK(f.registry.Publish("boot-A", r.region.id, "w0", 0, {"a"}).code);
    CHECK(f.registry.Leave("boot-A", "weights-A", "w0") ==
          PolicyRegionCode::BUSY);
    OK(f.registry.Close("boot-A", r.region.id, "w0"));
    auto v = f.registry.Acquire("boot-A", "weights-A", "w0", {"a"});
    OK(v.code);
    CHECK(f.registry.Leave("boot-A", "weights-A", "w0") ==
          PolicyRegionCode::BUSY);
    OK(f.registry.Release("boot-A", v.read_id, "w0"));
    for (int w = 0; w < 3; w++)
        OK(f.registry.Leave("boot-A", "weights-A", "w" + std::to_string(w)));
    auto live = f.registry.Acquire("boot-A", "weights-A", "w3", {"a"});
    OK(live.code);
    CHECK(live.objects[0].found);
    OK(f.registry.Join("boot-A", "weights-B", "w0"));
    OK(f.registry.Release("boot-A", live.read_id, "w3"));
    OK(f.registry.Leave("boot-A", "weights-A", "w3"));
    OK(f.registry.Leave("boot-A", "weights-A", "w3"));
    CHECK(f.registry.Join("boot-A", "weights-A", "w0") ==
          PolicyRegionCode::STALE);
    CHECK(f.registry.Acquire("boot-A", "weights-A", "w0", {"a"}).code ==
          PolicyRegionCode::STALE);
    f.registry.Collect();
    CHECK(f.freed == 1);
    OK(f.registry.Join("boot-A", "weights-B", "w1"));
}
void TestGcRetry() {
    int successful = 0, calls = 0;
    PolicyRegionRegistry r{
        "b",
        [](const auto&, uint64_t n) { return PolicyExtent{"p", 0, n, n, 0}; },
        [&](const auto&, const auto&) {
            if (++calls == 1) throw std::runtime_error("transient free");
            ++successful;
        }};
    auto a = r.Reserve("p", "w", "n", 1, 1, 1);
    OK(a.code);
    OK(r.Close("b", a.region.id, "w"));
    OK(r.Revoke("b", "p"));
    OK(r.Reclaim("b", "p"));
    r.Collect();
    CHECK(r.Stats().retained_bytes == 1 && successful == 0);
    r.Collect();
    CHECK(r.Stats().retained_bytes == 0 && successful == 1);
    r.Collect();
    CHECK(successful == 1);
}

int main() {
    int failures = 0;
    std::vector<std::pair<const char*, void (*)()>> tests = {
        {"geometry", TestGeometry},
        {"reservation-retry", TestReservationRetry},
        {"publication/read", TestPublicationAndRead},
        {"atomic-duplicates", TestAtomicDuplicateRejection},
        {"fence/drain/async-gc", TestFenceAndDrain},
        {"boot/closed-writer", TestBootAndClosedWriter},
        {"overlap/128-epochs", TestOverlapAndLongRun},
        {"allocation-failure/limits", TestAllocationFailureAndLimits},
        {"concurrent-owners", TestConcurrentOwners},
        {"participation/overlap", TestParticipation},
        {"gc-retry", TestGcRetry}};
    for (auto [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& e) {
            ++failures;
            std::cout << "FAIL " << name << ": " << e.what() << '\n';
        }
    }
    return failures ? 1 : 0;
}
