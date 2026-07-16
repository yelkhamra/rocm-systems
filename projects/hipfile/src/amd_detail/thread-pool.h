/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

namespace hipFile {

class ITaskGroup {
public:
    virtual ~ITaskGroup() = default;

    virtual void run(std::function<void()> work) = 0;
    virtual void cancel()                        = 0;
    virtual void wait()                          = 0;
};

class IThreadPool {
public:
    virtual ~IThreadPool() = default;

    virtual std::unique_ptr<ITaskGroup> makeTaskGroup() = 0;
};

class ThreadPool : public IThreadPool {
public:
    explicit ThreadPool(size_t workers = std::thread::hardware_concurrency());
    ~ThreadPool() override;

    ThreadPool(const ThreadPool &)            = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    ThreadPool(ThreadPool &&)            = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    std::unique_ptr<ITaskGroup> makeTaskGroup() override;

private:
    struct Impl;

    std::unique_ptr<Impl> impl;
};

HIPFILE_CONTEXT_DEFAULT_IMPL(IThreadPool, ThreadPool);

}
