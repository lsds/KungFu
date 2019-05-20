#pragma once
#include <condition_variable>
#include <mutex>
#include <numeric>

#include <gtest/gtest.h>

#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_types.hpp>

#include <iostream>

class Waiter
{
    std::mutex mu;
    std::condition_variable cv;
    bool _done;

  public:
    Waiter() : _done(false) {}

    void done()
    {
        std::lock_guard<std::mutex> lk(mu);
        _done = true;
        cv.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return this->_done; });
    }
};
