#include <cstdio>
#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_init.h>

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_tensorflow_init() { _kungfu_world.reset(new kungfu_world); }

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

}  // namespace kungfu
