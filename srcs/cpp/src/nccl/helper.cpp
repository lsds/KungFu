#include <numeric>
#include <sstream>

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

static std::string topology_key(const std::vector<int32_t> &topology)
{
    std::stringstream ss;
    for (size_t i = 0; i < topology.size(); ++i) {
        if (i > 0) { ss << ','; }
        ss << topology[i];
    }
    return ss.str();
}

NCCLController *NCCLHelper::EnsureGroupController(std::vector<int32_t> topology)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = group_controllers_[topology_key(topology)];
    if (ptr.get() == nullptr) {
        ptr.reset(new NCCLController(std::move(topology)));
    }
    return ptr.get();
}

std::unique_ptr<NCCLHelper> &NCCLHelper::GetDefault(bool reinit)
{
    static std::unique_ptr<NCCLHelper> instance;
    if (reinit) { instance.reset(new NCCLHelper); }
    return instance;
}
}  // namespace kungfu
