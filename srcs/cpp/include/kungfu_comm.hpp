#pragma once
#include <kungfu.h>
#include <kungfu_types.hpp>

namespace kungfu
{
// sync_communicator must be used with an order group
class sync_communicator
{
  public:
    virtual void bcast(const void *send_buf, void *recv_buf, size_t count,
                       KungFu_Datatype dtype) = 0;

    virtual void reduce(const void *send_buf, void *recv_buf, size_t count,
                        KungFu_Datatype dtype) = 0;

    virtual void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                            KungFu_Datatype dtype) = 0;
};

class async_communicator
{
  public:
    virtual void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                            KungFu_Datatype dtype, const char *name,
                            DoneCallback done) = 0;
};

extern sync_communicator *new_local_nccl_comm(kungfu_world &bootstrap);
extern async_communicator *new_inter_go_comm(kungfu_world &bootstrap);

}  // namespace kungfu
