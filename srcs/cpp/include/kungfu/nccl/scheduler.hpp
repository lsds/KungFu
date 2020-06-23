#pragma once
#include <map>
#include <memory>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/common.hpp>

namespace kungfu
{
// order_group wraps order_group_t
class order_group
{
    order_group_t *og_;
    std::map<std::string, int> ranks_;

  public:
    using Task = DoneCallback;

    order_group(const std::vector<std::string> &names,
                const std::vector<int32_t> &order);

    ~order_group();

    void Start(const std::string &name, const Task &task);

    std::vector<int32_t> Wait();
};

class NCCLScheduler
{
    const std::string name_;
    const bool auto_order_;
    const KungFu_NCCLScope scope_;

    int counter_;
    std::vector<int32_t> order_;

    std::unique_ptr<order_group> order_group_;

    void ResetOrder(int n);

  public:
    NCCLScheduler(const KungFu_NCCLScope scope, const bool auto_order = true);

    void Reset(const std::vector<std::string> &names);

    void Start(const std::string &name, const order_group::Task &task);
};
}  // namespace kungfu
