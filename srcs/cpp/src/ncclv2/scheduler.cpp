#include <iostream>

#include <kungfu/ncclv2/scheduler.hpp>

namespace kungfu
{
NCCLScheduler_V2::NCCLScheduler_V2(KungFu_NCCLScope scope, Peer *peer,
                                   NCCLController_V2 *controller)
    : auto_order_(false), scope_(scope), peer_(peer), controller_(controller),
      step_(0)
{
    nccl_thread_.reset(new std::thread([&] {
        DBG("nccl thread started");
        bool stopped            = false;
        bool last_step_finished = true;
        for (int i = 0; !stopped; ++i) {
            auto t = comitted_tasks_.get();
            switch (t.first) {
            case TASK_STOP: {
                if (!last_step_finished) {
                    throw std::runtime_error(
                        "can't begin step before last step finished");
                }
                stopped = true;
                break;
            }
            case TASK_BEGIN_STEP: {
                if (!last_step_finished) {
                    throw std::runtime_error(
                        "can't begin step before last step finished");
                }
                last_step_finished = false;
                break;
            }
            case TASK_END_STEP: {
                last_step_finished = true;
                break;
            }
            case TASK_OP:
                break;
            default:
                throw std::runtime_error("invalid task type");
            }
            auto *pt = t.second;
            if (pt != nullptr) {
                (*pt)();
                delete pt;
            }
        }
        DBG("nccl thread stopped");
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
    ++step_;
    comitted_tasks_.put(std::make_pair(
        TASK_BEGIN_STEP, new Task([&] { this->controller_->InitOnce(); })));
    this->names_ = names;
    ranks_.clear();
    const int n = names.size();
    for (int i = 0; i < n; ++i) { ranks_[names[i]] = i; }

    pending_tasks_.clear();
    pending_tasks_.resize(n);
    last_commit_ = -1;

    arrive_order_.clear();
    arrive_order_.reserve(n);
}

void NCCLScheduler_V2::EndStep()
{
    comitted_tasks_.put(std::make_pair(TASK_END_STEP, nullptr));
    if (auto_order_ && step_ == 1) {
        DBG("Broadcast order");
        using T = decltype(arrive_order_)::value_type;
        std::vector<T> common_order(arrive_order_.size());
        if (scope_ == KungFu_NCCL_LOCAL) {
            peer_->LocalBroadcast(arrive_order_.data(), common_order.data(),
                                  common_order.size(), type_encoder::value<T>(),
                                  "order");
        } else {
            peer_->Broadcast(arrive_order_.data(), common_order.data(),
                             common_order.size(), type_encoder::value<T>(),
                             "order");
        }

        DBG("Apply common order");
        const int n = names_.size();
        std::vector<std::string> names(n);
        for (int i = 0; i < n; ++i) { names[i] = names_[common_order[i]]; }
        ranks_.clear();
        for (int i = 0; i < n; ++i) { ranks_[names[i]] = i; }
    }
}

void NCCLScheduler_V2::Enqueue(const std::string &name,
                               std::function<void()> task)
{
    std::lock_guard<std::mutex> lk(mu_);

    const int rank = ranks_.at(name);
    if (step_ == 1) { arrive_order_.push_back(rank); }
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
