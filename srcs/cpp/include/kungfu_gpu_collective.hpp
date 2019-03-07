#pragma once
#include <kungfu.h>
#include <kungfu_types.hpp>

namespace kungfu
{
class gpu_collective
{
  public:
    virtual ~gpu_collective() {}

    virtual void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                            KungFu_Datatype dtype) = 0;

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count)
    {
        all_reduce(send_buf, recv_buf, count, type_encoder::value<T>());
    }
};

extern gpu_collective *new_gpu_collective(kungfu_world &world);
}  // namespace kungfu
