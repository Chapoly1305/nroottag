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
 
#ifndef PREFIX_THREAD_POOL_H
#define PREFIX_THREAD_POOL_H

#include <thread>
#include <atomic>
#include <vector>
#include <queue>
#include <condition_variable>
#include "Vanity.h"

class VanitySearch;

struct CheckPrefixesWorkItem {
    uint32_t startIdx;
    uint32_t endIdx;
    const CHECK_PREFIXES* result;
    std::vector<Int>* keys;
};

class PrefixThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<CheckPrefixesWorkItem> tasks;

    // Protects access to 'tasks' and signals changes to threads
    std::mutex queue_mutex;
    std::condition_variable condition;

    // Tracks whether we are shutting down
    std::atomic<bool> stop;

    // **NEW**: Tracks how many tasks are currently "in flight"
    std::atomic<size_t> inFlight{0};

    VanitySearch* vs;
    size_t numThreads;

    void workerThread();
    void processWorkItem(const CHECK_PREFIXES* result,
                         uint32_t idx,
                         std::vector<Int>& keys,
                         Point& temp);

public:
    PrefixThreadPool(VanitySearch* vs, size_t threads);
    ~PrefixThreadPool();

    void addTask(CheckPrefixesWorkItem&& work);

    // Wait until 'tasks.empty() && inFlight == 0'
    void waitForCompletion();

    size_t getNumThreads() const { return numThreads; }
};

#endif // PREFIX_THREAD_POOL_H
