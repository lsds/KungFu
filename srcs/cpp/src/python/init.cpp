#include <cstdio>
#include <numeric>

#include <kungfu.h>
#include <kungfu/python/init.h>
#include <kungfu/utils/trace.hpp>

DEFINE_TRACE_CONTEXT(kungfu);

std::unique_ptr<kungfu::Peer> _default_peer;

void kungfu_python_init() { _default_peer.reset(new kungfu::Peer); }

void kungfu_python_finialize() { _default_peer.reset(nullptr); }

uint64_t kungfu_uid() { return _default_peer->Uid(); }

int kungfu_detached() { return _default_peer->Detached(); }

int kungfu_rank() { return _default_peer->Rank(); }

int kungfu_size() { return _default_peer->Size(); }

int kungfu_local_rank() { return _default_peer->LocalRank(); }

int kungfu_local_size() { return _default_peer->LocalSize(); }

void kungfu_barrier() { _default_peer->Barrier(); }

int kungfu_propose_new_size(int new_size)
{
    return _default_peer->ProposeNewSize(new_size);
}

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
    if (auto_order_ && order_group_.get() != nullptr) {
        if (counter_ == 1) {
            using T                           = int32_t;
            const std::vector<T> arrive_order = order_group_->Wait();
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
    order_group_.reset(new order_group(names, order_));
    ++counter_;
}

void NCCLScheduler::Start(const std::string &name,
                          const order_group::Task &task)
{
    order_group_->Start(name, task);
}

const std::map<std::string, KungFu_NCCLScope> _nccl_scopes({
    {"global", KungFu_NCCL_GLOBAL},
    {"local", KungFu_NCCL_LOCAL},
});
}  // namespace kungfu
