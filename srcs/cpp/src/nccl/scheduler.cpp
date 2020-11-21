#include <numeric>

#include <kungfu/nccl/scheduler.hpp>
#include <kungfu/utils/waiter.hpp>

namespace kungfu
{
NCCLThread::NCCLThread()
{
    thread_.reset(new std::thread([&] {
        for (;;) {
            auto t = queue_.get();
            if (t == nullptr) { break; }
            t();
        }
    }));
}

NCCLThread::~NCCLThread()
{
    queue_.put(nullptr);
    thread_->join();
}

void NCCLThread::Put(std::function<void()> task) { queue_.put(task); }

void NCCLThread::Do(std::function<void()> task)
{
    Waiter waiter;
    queue_.put([=, waiter = &waiter] {
        task();
        waiter->done();
    });
    waiter.wait();
}

LinearExecutor::LinearExecutor(const std::vector<std::string> &names,
                               const std::vector<int32_t> &order,
                               NCCLThread *nccl_thread)
    : size_(names.size()), started_(0), arrive_order_(size_),
      is_started_(size_), tasks_(size_), nccl_thread_(nccl_thread)
{
    for (int i = 0; i < size_; ++i) { ranks_[names[order[i]]] = i; }
    std::fill(is_started_.begin(), is_started_.end(), false);
    executor_.reset(new std::thread([&] {
        for (int i = 0; i < size_; ++i) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return is_started_[i]; });
            }
            nccl_thread_->Put(tasks_[i]);
        }
        nccl_thread_->Do([] {});  // do an empty task to wait all previous tasks
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
      auto_order_(auto_order), scope_(scope), counter_(0),
      nccl_thread_(new NCCLThread)
{
}

void NCCLScheduler::ResetOrder(int n)
{
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), 0);
}

void NCCLScheduler::Reset(const std::vector<std::string> &names, Peer *peer)
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
                    peer->LocalBroadcast(
                        arrive_order.data(), order_.data(), order_.size(),
                        type_encoder::value<T>(), name_.c_str());
                } else {
                    peer->Broadcast(arrive_order.data(), order_.data(),
                                    order_.size(), type_encoder::value<T>(),
                                    name_.c_str());
                }
            }
        }
    }
    executor_.reset(new LinearExecutor(names, order_, nccl_thread_.get()));
    ++counter_;
}

void NCCLScheduler::Start(const std::string &name, const DoneCallback &task)
{
    executor_->Start(name, task);
}

void NCCLScheduler::Do(std::function<void()> task)
{
    nccl_thread_->Do(std::move(task));
}
}  // namespace kungfu
