#include <numeric>

#include <kungfu/nccl/scheduler.hpp>
#include <kungfu/python/init.h>  // FIXME: remove

namespace kungfu
{
order_group::order_group(const std::vector<std::string> &names,
                         const std::vector<int32_t> &order)
    : og_(new_ranked_order_group(names.size()))
{
    const int n = names.size();
    for (int i = 0; i < n; ++i) { ranks_[names[order[i]]] = i; }
}

order_group::~order_group()
{
    Wait();
    del_order_group(og_);
}

void order_group::Start(const std::string &name, const Task &task)
{
    order_group_do_rank(og_, ranks_.at(name), new CallbackWrapper(task));
}

std::vector<int32_t> order_group::Wait()
{
    std::vector<int32_t> arrive_order(ranks_.size());
    order_group_wait(og_, arrive_order.data());
    return arrive_order;
}

LinearExecutor::LinearExecutor(const std::vector<std::string> &names,
                               const std::vector<int32_t> &order)
    : size_(names.size()), started_(0), arrive_order_(size_),
      is_started_(size_), tasks_(size_)
{
    for (int i = 0; i < size_; ++i) { ranks_[names[order[i]]] = i; }
    std::fill(is_started_.begin(), is_started_.end(), false);
    executor_.reset(new std::thread([&] {
        for (int i = 0; i < size_; ++i) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return is_started_[i]; });
            }
            tasks_[i]();
        }
    }));
}

LinearExecutor::~LinearExecutor() { executor_->join(); }

void LinearExecutor::Start(const std::string &name, const DoneCallback &task)
{
    std::unique_lock<std::mutex> _lk(mu_);
    const int32_t rank      = ranks_.at(name);
    arrive_order_[started_] = rank;
    is_started_[rank]       = true;
    tasks_[rank]            = task;
    started_++;
    cv_.notify_all();
}

std::vector<int32_t> LinearExecutor::Wait()
{
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return started_ == size_; });
    }
    return arrive_order_;
}

NCCLScheduler::NCCLScheduler(const KungFu_NCCLScope scope,
                             const bool auto_order)
    : name_("NCCLScheduler_" + std::to_string(int(scope))),
      auto_order_(auto_order), scope_(scope), counter_(0)
{
}

void NCCLScheduler::ResetOrder(int n)
{
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), 0);
}

void NCCLScheduler::Reset(const std::vector<std::string> &names)
{
    if (names.size() != order_.size()) {
        // FIXME: also check value of names
        // FIXME: reset counter
        ResetOrder(names.size());
    }
    if (auto_order_ && executor_.get() != nullptr) {
        if (counter_ == 1) {
            using T                           = int32_t;
            const std::vector<T> arrive_order = executor_->Wait();
            if (arrive_order.size() == order_.size()) {
                if (scope_ == KungFu_NCCL_LOCAL) {
                    _default_peer->LocalBroadcast(
                        arrive_order.data(), order_.data(), order_.size(),
                        type_encoder::value<T>(), name_.c_str());
                } else {
                    _default_peer->Broadcast(
                        arrive_order.data(), order_.data(), order_.size(),
                        type_encoder::value<T>(), name_.c_str());
                }
            }
        }
    }
    executor_.reset(new LinearExecutor(names, order_));
    ++counter_;
}

void NCCLScheduler::Start(const std::string &name, const DoneCallback &task)
{
    executor_->Start(name, task);
}
}  // namespace kungfu
