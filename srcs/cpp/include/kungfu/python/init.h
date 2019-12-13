#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/gpu_collective.hpp>

extern "C" {
extern void kungfu_python_init();
extern void kungfu_python_init_gpu();

extern void kungfu_python_finialize();
extern void kungfu_python_finialize_gpu();

extern int kungfu_get_cuda_index();

// helpers APIs to access kungfu without tensorflow operators
// FIXME: don't mix with tensorflow binding
extern int kungfu_rank();          // get current rank
extern int kungfu_local_rank();    // get current local rank
extern int kungfu_cluster_size();  // get current size
extern void kungfu_barrier();
}

extern std::unique_ptr<kungfu_world> _kungfu_world;

namespace kungfu
{
// order_group wraps order_group_t
class order_group
{
    order_group_t *og_;
    std::map<std::string, int> ranks_;

  public:
    using Task = DoneCallback;

    order_group(const std::vector<std::string> &names);

    ~order_group();

    void Start(const std::string &name, const Task &task);

    void Wait();
};

extern std::unique_ptr<order_group> _nccl_order_group;

class nccl_controller
{
    std::unique_ptr<gpu_collective> _gpu_collective;

  public:
    nccl_controller();

    ~nccl_controller() = default;

    int ScheduledAllReduce(DoneCallback ready, const void *sendbuf,
                           void *recvbuf, int count, KungFu_Datatype dtype,
                           KungFu_Op op, const char *name, DoneCallback done);

    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name,
                  DoneCallback done);
};

extern std::unique_ptr<nccl_controller> _nccl_controller;
}  // namespace kungfu
