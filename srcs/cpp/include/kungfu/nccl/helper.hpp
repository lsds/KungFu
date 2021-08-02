#pragma once
#include <map>
#include <mutex>
#include <vector>

#include <kungfu.h>
#include <kungfu/cuda/stream.hpp>
#include <kungfu/nccl/common.hpp>
#include <kungfu/nccl/controller.hpp>
#include <kungfu/nccl/scheduler.hpp>

namespace kungfu
{
// NCCLHelper is a singleton class that contains NCCL related global variables
class NCCLHelper
{
    std::mutex mu_;

    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLScheduler>> schedulers_;
    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLController>> controllers_;

    std::map<std::string, std::unique_ptr<NCCLController>> group_controllers_;

  public:
    NCCLHelper();

    NCCLScheduler *EnsureScheduler(const KungFu_NCCLScope scope);

    NCCLController *EnsureController(const KungFu_NCCLScope scope);

    NCCLController *EnsureGroupController(std::vector<int32_t> topology);

    static std::unique_ptr<NCCLHelper> &GetDefault(bool reinit = false);
};
}  // namespace kungfu
