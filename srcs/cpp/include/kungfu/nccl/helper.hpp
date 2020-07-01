#pragma once
#include <map>
#include <mutex>

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

    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLController>> controllers_;
    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLScheduler>> schedulers_;

  public:
    NCCLHelper();

    NCCLController *EnsureController(const KungFu_NCCLScope scope);

    NCCLScheduler *EnsureScheduler(const KungFu_NCCLScope scope);
};
}  // namespace kungfu
