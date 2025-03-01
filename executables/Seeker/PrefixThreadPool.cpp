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
 
#include "PrefixThreadPool.h"
#include "Vanity.h"
#include <iostream>

PrefixThreadPool::PrefixThreadPool(VanitySearch* vs, size_t threads) 
    : stop(false), vs(vs), numThreads(threads) 
{
    printf("Starting thread pool with %zu threads\n", threads);
    // Launch worker threads
    for (size_t i = 0; i < threads; i++) {
        workers.emplace_back(&PrefixThreadPool::workerThread, this);
    }
}

PrefixThreadPool::~PrefixThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();

    // Join all worker threads
    for (std::thread& worker : workers) {
        worker.join();
    }
}

void PrefixThreadPool::addTask(CheckPrefixesWorkItem&& work) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(std::move(work));

        // **NEW**: Increment the "in flight" counter
        inFlight.fetch_add(1, std::memory_order_relaxed);
    }
    // Wake up one worker thread
    condition.notify_one();
}

void PrefixThreadPool::waitForCompletion() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    condition.wait(lock, [this] {
        // Only done when no tasks in queue AND nothing in flight
        return tasks.empty() && (inFlight.load(std::memory_order_relaxed) == 0);
    });
}

void PrefixThreadPool::processWorkItem(const CHECK_PREFIXES* result,
                                       uint32_t idx,
                                       std::vector<Int>& keys,
                                       Point& temp) 
{
    if (!result || !result->raw || idx >= result->size) {
        return;
    }

    // itemBase points to the start of the item
    uint32_t* itemBase = result->raw + (idx * ITEM_SIZE32);
    // Typically you skip the first 32-bit word if that's your layout
    uint32_t* itemPtr = itemBase + 1;

    ITEM it;
    it.thId = itemPtr[0];

    // Make sure thread ID is within range of 'keys'
    if (it.thId >= keys.size()) {
        return;
    }

    int16_t* ptr = (int16_t*)&(itemPtr[1]);
    it.endo = ptr[0] & 0x7FFF;
    it.mode = (ptr[0] & 0x8000) != 0;
    it.incr = ptr[1];
    it.hash = (uint8_t*)(itemPtr + 2);

    // Copy the 8-bit blocks into temp
    for (int x = 0; x < NB08BLOCK; x++) {
        temp.x.bits08[x] = it.hash[x];
    }

    // Actual check
    vs->checkPubKey(temp.x.bits16[13], keys[it.thId], it.incr, 0, temp);
}

void PrefixThreadPool::workerThread() {
    Point temp;

    while (true) {
        CheckPrefixesWorkItem work;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Wait until we have tasks or we are stopping
            condition.wait(lock, [this] { 
                return !tasks.empty() || stop.load(); 
            });

            // If we are shutting down and there are no tasks, exit thread
            if (stop && tasks.empty()) {
                return;
            }

            // We have something to do
            work = std::move(tasks.front());
            tasks.pop();
        }

        // Process the assigned range
        if (work.result && work.result->raw && work.keys &&
            work.startIdx < work.endIdx && work.endIdx <= work.result->size) 
        {
            for (uint32_t i = work.startIdx; i < work.endIdx; i++) {
                processWorkItem(work.result, i, *work.keys, temp);
            }
        }

        // **NEW**: After finishing, decrement "in flight" and notify
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            inFlight.fetch_sub(1, std::memory_order_relaxed);
            condition.notify_all();
        }
    }
}
