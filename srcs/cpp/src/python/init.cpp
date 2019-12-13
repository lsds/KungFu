#include <cstdio>

#include <kungfu.h>
#include <kungfu/python/init.h>

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_python_init() { _kungfu_world.reset(new kungfu_world); }

void kungfu_python_finialize() { _kungfu_world.reset(nullptr); }

int kungfu_rank() { return _kungfu_world->Rank(); }

int kungfu_local_rank() { return _kungfu_world->LocalRank(); }

int kungfu_cluster_size() { return _kungfu_world->ClusterSize(); }

void kungfu_barrier() { _kungfu_world->Barrier(); }

namespace kungfu
{
order_group::order_group(const std::vector<std::string> &names)
    : og_(new_ranked_order_group(names.size()))
{
    int idx = 0;
    for (const auto &name : names) { ranks_[name] = idx++; }
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

void order_group::Wait() { order_group_wait(og_); }

std::unique_ptr<order_group> _nccl_order_group;
}  // namespace kungfu
