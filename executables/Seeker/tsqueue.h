/*
 * Copyright (c) 2025 Chapoly1305
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef VANITYSEARCH_TSQUEUE_H
#define VANITYSEARCH_TSQUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <atomic>
#include <vector>
#include <thread>

// Simple spin-wait implementation without intrinsics
class SpinWait {
private:
    static constexpr int MAX_SPIN = 64;
    int count = 0;

public:
    void spin() noexcept {
        if (count < MAX_SPIN) {
            for (int i = 0; i < (1 << count); ++i) {
                std::atomic_signal_fence(std::memory_order_relaxed);
            }
            ++count;
        } else {
            std::this_thread::yield();
        }
    }

    void reset() noexcept {
        count = 0;
    }
};

template <typename T>
class TSQueue {
private:
    // Cacheline size on most x86 processors
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    // Align structures to prevent false sharing
    struct alignas(CACHE_LINE_SIZE) AlignedAtomic {
        std::atomic<size_t> value;
        AlignedAtomic() : value(0) {}
        char padding[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)];
    };

    // Constants tuned for GPU workload
    static constexpr size_t MAX_QUEUE_SIZE = 256;
    static constexpr size_t SPIN_ATTEMPTS = 1000;
    
    // Core queue components
    std::queue<T> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_notEmpty;
    std::condition_variable m_notFull;
    
    // Aligned atomic counters to prevent false sharing
    AlignedAtomic m_size;
    AlignedAtomic m_pushes;
    AlignedAtomic m_pops;
    
    // Shutdown flag
    std::atomic<bool> _shutdown{false};

    // Thread local spin wait
    thread_local static SpinWait spinWait;

public:
    TSQueue() = default;
    ~TSQueue() { shutdown(); }

    // Deleted copy/move operations for safety
    TSQueue(const TSQueue&) = delete;
    TSQueue& operator=(const TSQueue&) = delete;
    TSQueue(TSQueue&&) = delete;
    TSQueue& operator=(TSQueue&&) = delete;

    size_t size() const noexcept {
        return m_size.value.load(std::memory_order_relaxed);
    }

    void push(T&& item) {
        // Fast path: try spinning first
        for (size_t i = 0; i < SPIN_ATTEMPTS; ++i) {
            std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
            if (lock && m_queue.size() < MAX_QUEUE_SIZE) {
                m_queue.push(std::move(item));
                m_size.value.fetch_add(1, std::memory_order_relaxed);
                m_pushes.value.fetch_add(1, std::memory_order_relaxed);
                lock.unlock();
                m_notEmpty.notify_one();
                spinWait.reset();
                return;
            }
            spinWait.spin();
        }

        // Slow path: wait on condition variable
        std::unique_lock<std::mutex> lock(m_mutex);
        while (!_shutdown && m_queue.size() >= MAX_QUEUE_SIZE) {
            m_notFull.wait(lock);
        }
        
        if (_shutdown) return;
        
        m_queue.push(std::move(item));
        m_size.value.fetch_add(1, std::memory_order_relaxed);
        m_pushes.value.fetch_add(1, std::memory_order_relaxed);
        lock.unlock();
        m_notEmpty.notify_one();
    }

    bool pop(T& value) {
        // Fast path: try spinning first
        for (size_t i = 0; i < SPIN_ATTEMPTS; ++i) {
            std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
            if (lock && !m_queue.empty()) {
                value = std::move(m_queue.front());
                m_queue.pop();
                m_size.value.fetch_sub(1, std::memory_order_relaxed);
                m_pops.value.fetch_add(1, std::memory_order_relaxed);
                lock.unlock();
                m_notFull.notify_one();
                spinWait.reset();
                return true;
            }
            spinWait.spin();
        }

        // Slow path: wait on condition variable
        std::unique_lock<std::mutex> lock(m_mutex);
        while (!_shutdown && m_queue.empty()) {
            m_notEmpty.wait(lock);
        }
        
        if (_shutdown && m_queue.empty()) return false;
        
        value = std::move(m_queue.front());
        m_queue.pop();
        m_size.value.fetch_sub(1, std::memory_order_relaxed);
        m_pops.value.fetch_add(1, std::memory_order_relaxed);
        lock.unlock();
        m_notFull.notify_one();
        return true;
    }

    // Non-blocking try_pop
    bool try_pop(T& value) noexcept {
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        if (!lock || m_queue.empty()) {
            spinWait.spin();
            return false;
        }

        value = std::move(m_queue.front());
        m_queue.pop();
        m_size.value.fetch_sub(1, std::memory_order_relaxed);
        m_pops.value.fetch_add(1, std::memory_order_relaxed);
        spinWait.reset();
        
        lock.unlock();
        m_notFull.notify_one();
        return true;
    }

    void shutdown() noexcept {
        _shutdown = true;
        m_notEmpty.notify_all();
        m_notFull.notify_all();
    }

    bool done() const noexcept {
        return _shutdown && m_size.value.load(std::memory_order_relaxed) == 0;
    }

    // Performance metrics
    struct QueueMetrics {
        size_t currentSize;
        size_t totalPushes;
        size_t totalPops;
    };

    QueueMetrics getMetrics() const noexcept {
        return {
            m_size.value.load(std::memory_order_relaxed),
            m_pushes.value.load(std::memory_order_relaxed),
            m_pops.value.load(std::memory_order_relaxed)
        };
    }
};

// Initialize thread_local spin wait
template <typename T>
thread_local SpinWait TSQueue<T>::spinWait;

#endif // VANITYSEARCH_TSQUEUE_H