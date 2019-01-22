#pragma once
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>

#include "trace.hpp"

#include <kungfu.h>
#include <kungfu_types.hpp>

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

std::string safe_getenv(const char *name)
{
    const char *ptr = std::getenv(name);
    if (ptr) { return std::string(ptr); }
    return "";
}

int getTestClusterSize()
{
    return std::stoi(safe_getenv("KUNGFU_TEST_CLUSTER_SIZE"));
}

int getSelfRank() { return std::stoi(safe_getenv("KUNGFU_SELF_RANK")); }
