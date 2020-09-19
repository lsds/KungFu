#include <numeric>

#include <kungfu/nccl/helper.hpp>

namespace kungfu
{
NCCLHelper::NCCLHelper() {}

NCCLController *NCCLHelper::EnsureController(const KungFu_NCCLScope scope)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = controllers_[scope];
    if (ptr.get() == nullptr) { ptr.reset(new NCCLController(scope)); }
    return ptr.get();
}

NCCLScheduler *NCCLHelper::EnsureScheduler(const KungFu_NCCLScope scope)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = schedulers_[scope];
    if (ptr.get() == nullptr) { ptr.reset(new NCCLScheduler(scope)); }
    return ptr.get();
}
}  // namespace kungfu
