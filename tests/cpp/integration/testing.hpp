#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>

#include <kungfu.h>
#include <kungfu/utils/trace.hpp>

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

namespace testing
{
using clock_t    = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double>;
using instant_t  = std::chrono::time_point<clock_t>;

instant_t now() { return clock_t::now(); }

duration_t since(const instant_t &t0) { return now() - t0; }
}  // namespace testing
