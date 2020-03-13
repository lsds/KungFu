#include <cstdio>

#include <kungfu.h>
#include <kungfu/python/init.h>
#include <kungfu/utils/trace.hpp>

DEFINE_TRACE_CONTEXT(kungfu);

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_python_init() { _kungfu_world.reset(new kungfu_world); }

void kungfu_python_finialize() { _kungfu_world.reset(nullptr); }

uint64_t kungfu_uid() { return _kungfu_world->Uid(); }

int kungfu_rank() { return _kungfu_world->Rank(); }

int kungfu_local_rank() { return _kungfu_world->LocalRank(); }

int kungfu_cluster_size() { return _kungfu_world->ClusterSize(); }

void kungfu_barrier() { _kungfu_world->Barrier(); }

int kungfu_propose_new_size(int new_size)
{
    return _kungfu_world->ProposeNewSize(new_size);
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

std::unique_ptr<order_group> _nccl_order_group;
}  // namespace kungfu
