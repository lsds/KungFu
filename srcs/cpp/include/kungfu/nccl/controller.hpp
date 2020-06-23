#pragma once
#include <memory>

#include <kungfu/nccl/common.hpp>
#include <kungfu/nccl/gpu_collective.hpp>

namespace kungfu
{
class NCCLController
{
    KungFu_NCCLScope scope_;
    std::unique_ptr<gpu_collective> gpu_collective_;

  public:
    NCCLController(const KungFu_NCCLScope scope);

    void InitOnce();

    int Reduce(const Workspace &w, KungFu_Op op, DoneCallback done);

    int Broadcast(const Workspace &w, DoneCallback done);

    int AllReduce(const Workspace &w, KungFu_Op op, DoneCallback done);
};

void CrossAllReduceGpu(const Workspace &w, KungFu_Op op,
                       const std::string &name, DoneCallback done);
}  // namespace kungfu
