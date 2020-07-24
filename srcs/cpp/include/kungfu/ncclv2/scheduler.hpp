#pragma once
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <kungfu/nccl/common.hpp>
#include <kungfu/ncclv2/controller.hpp>
#include <kungfu/utils/channel.hpp>

namespace kungfu
{
enum TaskType {
    TASK_STOP,
    TASK_BEGIN_STEP,
    TASK_OP,
    TASK_END_STEP,
};

class NCCLScheduler_V2
{
    NCCLController_V2 *controller_;
    int step_;

    using Task      = std::function<void()>;
    using TaskQueue = MpscChannel<std::pair<TaskType, Task *>>;

    TaskQueue comitted_tasks_;

    std::mutex mu_;

    std::vector<std::string> names_;
    std::map<std::string, int> ranks_;

    std::vector<std::unique_ptr<Task>> pending_tasks_;
    int last_commit_;

    std::unique_ptr<std::thread> nccl_thread_;

  public:
    NCCLScheduler_V2(NCCLController_V2 *controller);
    ~NCCLScheduler_V2();

    void BeginStep(const std::vector<std::string> &names);
    void Enqueue(const std::string &name, std::function<void()> task);
    void EndStep();  // automatically called on the last Enqueue

    // void EnqueueInit(std::function<void()> task);
};
}  // namespace kungfu
