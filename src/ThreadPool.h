#pragma once

// Small condition-variable thread pool. Mirrors CuRast's
// `ThreadPool` ([Resources/CuRast/src/ThreadPool.h]) so the loader's
// per-accessor parallel processing matches the reference.
//
//   ThreadPool pool(N);
//   for (auto& task : tasks) pool.enqueue([&](int threadIdx){ ... });
//   pool.wait();
//
// The lambda gets its worker threadIndex so it can pick up a per-thread
// staging buffer. wait() blocks until activeTasks reaches zero.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace metalrast {

class ThreadPool {
public:
    int numThreads = 0;

    explicit ThreadPool(size_t n) : stop_(false), activeTasks_(0) {
        numThreads = static_cast<int>(n);
        workers_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this, i] {
                for (;;) {
                    std::function<void(int)> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex_);
                        cv_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task(static_cast<int>(i));
                    if (--activeTasks_ == 0) {
                        std::unique_lock<std::mutex> lk(waitMutex_);
                        waitCv_.notify_all();
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(queueMutex_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    void enqueue(std::function<void(int)> task) {
        { std::unique_lock<std::mutex> lock(queueMutex_);
          tasks_.push(std::move(task));
          ++activeTasks_; }
        cv_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(waitMutex_);
        waitCv_.wait(lock, [this]{ return activeTasks_ == 0; });
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    std::vector<std::thread>            workers_;
    std::queue<std::function<void(int)>> tasks_;
    std::mutex                          queueMutex_;
    std::condition_variable             cv_;
    std::mutex                          waitMutex_;
    std::condition_variable             waitCv_;
    std::atomic<size_t>                 activeTasks_;
    bool                                stop_;
};

}  // namespace metalrast
