#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace kungfu
{
class TensorBuffer
{
  public:
    TensorBuffer(size_t size) : size_(size), data_(new char[size]) { clear(); }

    size_t size() const { return size_; }

    void *data() const { return data_.get(); }

    void clear() { std::memset(data_.get(), 0, size_); }

  private:
    const size_t size_;
    std::unique_ptr<char[]> data_;
};

struct PushRequest {
    std::string name;
    std::uint32_t size;
    const void *data;
};

struct PullRequest {
};

template <typename T, typename R> void sendTo(const T &msg, R &w)
{
    // TODO: implement
}
}  // namespace kungfu
