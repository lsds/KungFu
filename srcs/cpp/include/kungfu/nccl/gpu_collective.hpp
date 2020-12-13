#pragma once
#include <kungfu.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void kungfu_show_cuda_version();
extern void kungfu_show_nccl_version();

#ifdef __cplusplus
}
#endif

namespace kungfu
{
class gpu_collective
{
  public:
    virtual ~gpu_collective() = default;

    virtual void reduce(const void *send_buf, void *recv_buf, size_t count,
                        KungFu_Datatype dtype) = 0;

    virtual void broadcast(const void *send_buf, void *recv_buf, size_t count,
                           KungFu_Datatype dtype) = 0;

    virtual void broadcast(const void *send_buf, void *recv_buf, size_t count,
                           KungFu_Datatype dtype, void *stream_ptr) = 0;

    virtual void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                            KungFu_Datatype dtype, KungFu_Op op) = 0;

    virtual void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            void *stream_ptr) = 0;

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count)
    {
        all_reduce(send_buf, recv_buf, count, type_encoder::value<T>());
    }
};

extern gpu_collective *new_global_gpu_collective(kungfu::Peer &);
extern gpu_collective *new_local_gpu_collective(kungfu::Peer &);
}  // namespace kungfu
