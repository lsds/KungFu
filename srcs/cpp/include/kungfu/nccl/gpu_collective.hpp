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
// gpu_collective wraps NCCL APIs for internal use.
// User should use NCCLController
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

    virtual void all_gather(const void *send_buf, void *recv_buf,
                            size_t send_count, KungFu_Datatype dtype) = 0;

    virtual void all_gather(const void *send_buf, void *recv_buf,
                            size_t send_count, KungFu_Datatype dtype,
                            void *stream_ptr) = 0;

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count)
    {
        all_reduce(send_buf, recv_buf, count, type_encoder::value<T>());
    }

    static gpu_collective *new_global(kungfu::Peer &);
    static gpu_collective *new_local(kungfu::Peer &);
    static gpu_collective *new_group(kungfu::Peer &,
                                     const std::vector<int32_t> &topology);
};
}  // namespace kungfu
