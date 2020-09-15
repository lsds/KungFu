#pragma once
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/common.hpp>
#include <kungfu/utils/channel.hpp>

namespace kungfu
{
// order_group wraps order_group_t
/*
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
*/

class LinearExecutor
{
    const int size_;

    std::mutex mu_;
    std::condition_variable cv_;

    std::map<std::string, int> ranks_;
    int started_;

    std::vector<int32_t> arrive_order_;
    std::vector<bool> is_started_;
    std::vector<DoneCallback> tasks_;
    std::unique_ptr<std::thread> executor_;

  public:
    LinearExecutor(const std::vector<std::string> &names,
                   const std::vector<int32_t> &order);

    ~LinearExecutor();

    void Start(const std::string &name, const DoneCallback &task);

    std::vector<int32_t> Wait();
};

class NCCLThread
{
    using Task      = std::function<void()>;
    using TaskQueue = MpscChannel<Task>;

    TaskQueue queue_;
    std::unique_ptr<std::thread> thread_;

  public:
    NCCLThread();
    ~NCCLThread();
    void Do(std::function<void()> task);
};

class NCCLScheduler
{
    const std::string name_;
    const bool auto_order_;
    const KungFu_NCCLScope scope_;

    int counter_;
    std::vector<int32_t> order_;

    // std::unique_ptr<order_group> order_group_;
    std::unique_ptr<NCCLThread> nccl_thread_;
    std::unique_ptr<LinearExecutor> executor_;

    void ResetOrder(int n);

  public:
    NCCLScheduler(const KungFu_NCCLScope scope, const bool auto_order = true);

    void Reset(const std::vector<std::string> &names);

    void Start(const std::string &name, const DoneCallback &task);

    // Run a task in the dedicated NCCL thread
    void Do(std::function<void()> task);
};
}  // namespace kungfu
