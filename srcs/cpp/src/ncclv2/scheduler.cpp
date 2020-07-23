#include <iostream>

#include <kungfu/ncclv2/scheduler.hpp>

namespace kungfu
{
NCCLScheduler_V2::NCCLScheduler_V2(NCCLController_V2 *controller)
    : controller_(controller)
{
    nccl_thread_.reset(new std::thread([&] {
        // DBG("nccl thread started");
        for (int i = 0;; ++i) {
            auto t = comitted_tasks_.get();
            if (t == nullptr) { break; }
            // DBG("nccl task started " + std::to_string(i));
            (*t)();
            delete t;
            // DBG("nccl task finished " + std::to_string(i));
        }
    }));
}

NCCLScheduler_V2::~NCCLScheduler_V2()
{
    comitted_tasks_.put(nullptr);
    nccl_thread_->join();
}

void NCCLScheduler_V2::BeginStep(const std::vector<std::string> &names)
{
    // DBG(__func__);
    comitted_tasks_.put(new Task([&] { this->controller_->InitOnce(); }));
    this->names_ = names;
    ranks_.clear();
    const int n = names.size();
    for (int i = 0; i < n; ++i) { ranks_[names[i]] = i; }

    pending_tasks_.clear();
    pending_tasks_.resize(n);
    last_commit_ = -1;
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
            // LOG_THREAD("committing " + std::to_string(i));
            comitted_tasks_.put(t.release());
            ++last_commit_;
        } else {
            break;
        }
    }
}
}  // namespace kungfu
