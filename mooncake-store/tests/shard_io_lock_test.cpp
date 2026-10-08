#include <gtest/gtest.h>

#include <chrono>
#include <future>

#include "storage/distributed/shard_io_lock.h"

namespace mooncake::test {
using namespace std::chrono_literals;

TEST(ShardIoLockTest, DisjointReadDoesNotWaitForWrite) {
    ShardIoLock lock;
    std::future<bool> read;
    {
        auto write = lock.Lock({{0, 4096, true}});
        read = std::async(std::launch::async, [&] {
            auto guard = lock.Lock({{4096, 8192, false}});
            return true;
        });
        EXPECT_EQ(read.wait_for(1s), std::future_status::ready);
    }
    EXPECT_TRUE(read.get());
}

TEST(ShardIoLockTest, OverlappingReadsCanProceed) {
    ShardIoLock lock;
    std::future<bool> second;
    {
        auto first = lock.Lock({{0, 8192, false}});
        second = std::async(std::launch::async, [&] {
            auto guard = lock.Lock({{4096, 8192, false}});
            return true;
        });
        EXPECT_EQ(second.wait_for(1s), std::future_status::ready);
    }
    EXPECT_TRUE(second.get());
}

TEST(ShardIoLockTest, OverlappingWriteWaitsForReaderCompletion) {
    ShardIoLock lock;
    std::future<bool> writer;
    std::promise<void> started;
    {
        auto reader = lock.Lock({{0, 8192, false}});
        writer = std::async(std::launch::async, [&] {
            started.set_value();
            auto guard = lock.Lock({{4096, 12288, true}});
            return true;
        });
        started.get_future().wait();
        EXPECT_EQ(writer.wait_for(50ms), std::future_status::timeout);
    }
    EXPECT_EQ(writer.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(writer.get());
}

TEST(ShardIoLockTest, BatchOwnsAllRangesAndMoveKeepsOwnership) {
    ShardIoLock lock;
    std::future<bool> writer;
    {
        auto original = lock.Lock({{0, 4096, true}, {8192, 12288, true}});
        auto moved = std::move(original);
        writer = std::async(std::launch::async, [&] {
            auto guard = lock.Lock({{8192, 12288, true}});
            return true;
        });
        EXPECT_EQ(writer.wait_for(50ms), std::future_status::timeout);
    }
    EXPECT_EQ(writer.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(writer.get());
}

TEST(ShardIoLockTest, DisjointWritesRemainSerialized) {
    ShardIoLock lock;
    std::future<void> writer;
    {
        auto first = lock.Lock({{0, 4096, true}});
        std::promise<void> started;
        writer = std::async(std::launch::async, [&] {
            started.set_value();
            auto second = lock.Lock({{8192, 12288, true}});
        });
        started.get_future().wait();
        EXPECT_EQ(writer.wait_for(std::chrono::milliseconds(50)),
                  std::future_status::timeout);
    }
    EXPECT_EQ(writer.wait_for(std::chrono::seconds(1)),
              std::future_status::ready);
    writer.get();
}

}  // namespace mooncake::test
