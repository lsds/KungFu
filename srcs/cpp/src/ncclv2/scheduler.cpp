#include <iostream>

#include <kungfu/ncclv2/scheduler.hpp>

namespace kungfu
{
NCCLScheduler_V2::NCCLScheduler_V2(NCCLController_V2 *controller)
    : controller_(controller), step_(0)
{
    nccl_thread_.reset(new std::thread([&] {
        DBG("nccl thread started");
        int step     = 0;
        int step_ops = 0;
        for (int i = 0;; ++i) {
            auto t = comitted_tasks_.get();
            switch (t.first) {
            case TASK_STOP:
                DBG("nccl thread stopped");
                return;
            case TASK_BEGIN_STEP: {
                ++step;
                DBG("nccl step started " + std::to_string(step));
                auto *pt = t.second;
                (*pt)();
                delete pt;
                DBG("resetting step_ops to ZERO from " +
                    std::to_string(step_ops));
                step_ops = 0;
                break;
            }
            case TASK_END_STEP: {
                DBG("nccl step finished " + std::to_string(step) +
                    " after ran " + std::to_string(step_ops));
                break;
            }
            case TASK_OP: {
                auto *pt = t.second;
                (*pt)();
                delete pt;
                ++step_ops;
                break;
            }
            default:
                throw std::runtime_error("invalid task type");
            }
        }
    }));
}

NCCLScheduler_V2::~NCCLScheduler_V2()
{
    comitted_tasks_.put(std::make_pair(TASK_STOP, nullptr));
    nccl_thread_->join();
}

void NCCLScheduler_V2::BeginStep(const std::vector<std::string> &names)
{
    // DBG(__func__);
    comitted_tasks_.put(std::make_pair(
        TASK_BEGIN_STEP, new Task([&] { this->controller_->InitOnce(); })));
    this->names_ = names;
    ranks_.clear();
    const int n = names.size();
    for (int i = 0; i < n; ++i) { ranks_[names[i]] = i; }

    pending_tasks_.clear();
    pending_tasks_.resize(n);
    last_commit_ = -1;
}

void NCCLScheduler_V2::EndStep()
{
    ++step_;
    comitted_tasks_.put(std::make_pair(TASK_END_STEP, nullptr));
}

void NCCLScheduler_V2::Enqueue(const std::string &name,
                               std::function<void()> task)
{
    std::lock_guard<std::mutex> lk(mu_);

    const int rank = ranks_.at(name);
    pending_tasks_[rank].reset(new Task(task));
    const int n = names_.size();
    for (int i = last_commit_ + 1; i < n; i++) {
        auto &t = pending_tasks_[i];
        if (t.get()) {
            // LOG_THREAD("committing " + std::to_string(i) + " : " + name);
            comitted_tasks_.put(std::make_pair(TASK_OP, t.release()));
            ++last_commit_;
        } else {
            break;
        }
    }

    if (last_commit_ + 1 == n) { EndStep(); }
}
}  // namespace kungfu
