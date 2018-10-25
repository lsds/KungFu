#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#ifdef KUNGFU_USER_LIBNOP
#include <nop/serializer.h>
#include <nop/utility/stream_writer.h>
#endif

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

#ifdef KUNGFU_USER_LIBNOP
    NOP_STRUCTURE(PushRequest, name, size);
#endif
};

struct PullRequest {
};

#ifdef KUNGFU_USER_LIBNOP

template <typename T, typename R> void sendTo(const T &msg, R &w)
{
    using Writer = nop::StreamWriter<R>;
    nop::Serializer<Writer> serializer;
    serializer.Write(msg);
}
#else
template <typename T, typename R> void sendTo(const T &msg, R &w) {}
#endif
}  // namespace kungfu
