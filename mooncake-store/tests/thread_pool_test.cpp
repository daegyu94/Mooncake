#include <gtest/gtest.h>
#include <glog/logging.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include "thread_pool.h"

#ifdef __linux__
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <system_error>

namespace {
thread_local int fail_on_create = 0;
thread_local int create_calls = 0;
thread_local int joined_threads = 0;
}  // namespace

// Interpose only in this test executable; libstdc++ calls pthread_create too.
extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                              void* (*start)(void*), void* arg) {
    if (fail_on_create && ++create_calls == fail_on_create) return EAGAIN;
    using Create = decltype(&pthread_create);
    static auto real_create =
        reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
    return real_create(thread, attr, start, arg);
}

extern "C" int pthread_join(pthread_t thread, void** result) {
    using Join = decltype(&pthread_join);
    static auto real_join =
        reinterpret_cast<Join>(dlsym(RTLD_NEXT, "pthread_join"));
    int error = real_join(thread, result);
    if (!error) ++joined_threads;
    return error;
}

#endif

namespace mooncake {

class ThreadPoolTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("ThreadPoolTest");
        FLAGS_logtostderr = 1;
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }
};

// Test basic task execution
TEST_F(ThreadPoolTest, BasicTaskExecution) {
    ThreadPool pool(2);
    std::atomic<int> counter(0);
    std::mutex mtx;
    std::condition_variable cv;
    int completed = 0;
    const int total_tasks = 10;

    for (int i = 0; i < total_tasks; ++i) {
        pool.enqueue([&]() {
            counter++;
            std::lock_guard<std::mutex> lock(mtx);
            completed++;
            cv.notify_one();
        });
    }

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]() { return completed == total_tasks; });

    EXPECT_EQ(counter.load(), total_tasks);
}

// Test parallel execution
TEST_F(ThreadPoolTest, ParallelExecution) {
    const size_t num_threads = 4;
    ThreadPool pool(num_threads);
    std::atomic<int> running_threads(0);
    std::atomic<int> max_concurrent_threads(0);
    std::mutex mtx;
    std::condition_variable cv;
    int completed = 0;
    const int total_tasks = 20;

    for (int i = 0; i < total_tasks; ++i) {
        pool.enqueue([&]() {
            int current = ++running_threads;
            int old_max = max_concurrent_threads.load();
            while (old_max < current &&
                   !max_concurrent_threads.compare_exchange_weak(old_max,
                                                                 current)) {
                // Keep trying to update max
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            --running_threads;

            std::lock_guard<std::mutex> lock(mtx);
            completed++;
            cv.notify_one();
        });
    }

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]() { return completed == total_tasks; });

    EXPECT_GE(max_concurrent_threads.load(), 1);
    EXPECT_LE(max_concurrent_threads.load(), num_threads);
}

// Test proper stop
TEST_F(ThreadPoolTest, ProperStop) {
    ThreadPool pool(2);
    std::atomic<int> counter(0);
    std::mutex mtx;
    std::condition_variable cv;
    int completed = 0;
    const int total_tasks = 5;

    for (int i = 0; i < total_tasks; ++i) {
        pool.enqueue([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter++;
            std::lock_guard<std::mutex> lock(mtx);
            completed++;
            cv.notify_one();
        });
    }

    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() { return completed == total_tasks; });
    }

    pool.stop();

    EXPECT_EQ(counter.load(), total_tasks);
    EXPECT_THROW({ pool.enqueue([]() {}); }, std::runtime_error);
}

// Test stress with many tasks
TEST_F(ThreadPoolTest, StressTest) {
    const int num_tasks = 1000;
    ThreadPool pool(4);
    std::atomic<int> counter(0);
    std::mutex mtx;
    std::condition_variable cv;
    int completed = 0;

    for (int i = 0; i < num_tasks; ++i) {
        pool.enqueue([&]() {
            counter++;
            std::lock_guard<std::mutex> lock(mtx);
            completed++;
            cv.notify_one();
        });
    }

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]() { return completed == num_tasks; });

    EXPECT_EQ(counter.load(), num_tasks);
}

#ifdef __linux__
namespace {

class ThreadPoolStartupTest : public ::testing::TestWithParam<int> {};

TEST_P(ThreadPoolStartupTest, JoinsStartedWorkersAndRethrowsCreationError) {
    const int failed_create = GetParam();
    // Isolate a broken constructor's hang/terminate and bound its runtime.
    ASSERT_EXIT(
        {
            alarm(5);
            fail_on_create = failed_create;
            create_calls = 0;
            joined_threads = 0;
            bool caught = false;
            try {
                ThreadPool pool(4);
            } catch (const std::system_error& error) {
                caught =
                    error.code() == std::errc::resource_unavailable_try_again;
            }
            if (!caught || create_calls != failed_create ||
                joined_threads != failed_create - 1) {
                std::_Exit(1);
            }

            fail_on_create = 0;
            std::atomic<int> completed{0};
            {
                ThreadPool pool(2);
                pool.enqueue([&] { ++completed; });
                pool.stop();
                pool.stop();
            }
            std::_Exit(completed == 1 && joined_threads == failed_create + 1
                           ? EXIT_SUCCESS
                           : EXIT_FAILURE);
        },
        ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

INSTANTIATE_TEST_SUITE_P(CreationFailures, ThreadPoolStartupTest,
                         ::testing::Values(1, 2, 4));

}  // namespace
#endif

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
