#pragma once

#include <functional>
#include <mutex>
#include <vector>
#include <queue>
#include <iostream>

class ThreadPool {
public:
    void Start();
    void QueueJob(const std::function<void()>& job);
    void Stop();
    bool QueueIsEmpty();
    int QueueSize();

private:
    void ThreadLoop();

    bool should_terminate = false;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};

void ThreadPool::Start() {
    const auto num_threads = std::thread::hardware_concurrency();
    std::cerr <<  "Number of threads: " << num_threads << std::endl;
    threads.resize(num_threads);
    for (auto i = 0; i < num_threads; i++) {
        threads.at(i) = std::thread(&ThreadPool::ThreadLoop, this);
    }
}

void ThreadPool::ThreadLoop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !jobs.empty() || should_terminate;
            });
            if (should_terminate) 
                return;

            job = jobs.front();
            jobs.pop();
        }
        job();
    }
}

void ThreadPool::QueueJob(const std::function<void()>& job) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
    }
    mutex_condition.notify_one();
}

void ThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();   // Unused threads should be terminated immediately

    // Wait for the threads that are still active to join
    for (std::thread& active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
}

bool ThreadPool::QueueIsEmpty() {
    return jobs.empty();
}

int ThreadPool::QueueSize() {
    return jobs.size();
}