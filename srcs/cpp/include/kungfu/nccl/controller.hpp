#pragma once
#include <memory>
#include <vector>

#include <kungfu/nccl/common.hpp>
#include <kungfu/nccl/gpu_collective.hpp>

namespace kungfu
{
void CrossAllReduceGpu(Peer *peer, const Workspace &w, KungFu_Op op,
                       const std::string &name, DoneCallback done);

// NCCLController exposes user-facing APIs
class NCCLController
{
    KungFu_NCCLScope scope_;

    // only used when scope_ == KungFu_NCCL_GROUP
    std::vector<int32_t> topology_;

    std::unique_ptr<gpu_collective> gpu_collective_;

    gpu_collective *new_gpu_collective(Peer *peer);

  public:
    NCCLController(const KungFu_NCCLScope scope);

    NCCLController(std::vector<int32_t> topology);

    void InitOnce(Peer *peer);

    void ReInit(Peer *peer);

    int Reduce(const Workspace &w, KungFu_Op op, DoneCallback done);

    int Broadcast(const Workspace &w, DoneCallback done);
    int Broadcast(const Workspace &w, void *stream_ptr);

    int AllReduce(const Workspace &w, KungFu_Op op, DoneCallback done);
    int AllReduce(const Workspace &w, KungFu_Op op, void *stream_ptr);

    int AllGather(const Workspace &w, DoneCallback done);
    int AllGather(const Workspace &w, void *stream_ptr);
};
}  // namespace kungfu
