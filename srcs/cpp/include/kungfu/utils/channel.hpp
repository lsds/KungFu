#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

namespace kungfu
{
template <typename T>
class MpscChannel  // unbound multiple producer single consumer channel
{
    std::mutex mu_;
    std::queue<T> buffer_;

    std::condition_variable cv_;

  public:
    T get()
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]() { return buffer_.size() > 0; });
        const T x = buffer_.front();
        buffer_.pop();
        return x;
    }

    void put(T x)
    {
        std::unique_lock<std::mutex> lk(mu_);
        buffer_.push(x);
        cv_.notify_one();
    }
};
}  // namespace kungfu
