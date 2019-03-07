#include <cstdio>
#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_init.h>

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_tensorflow_init() { _kungfu_world.reset(new kungfu_world); }

namespace kungfu
{
namespace tensorflow
{

all_reduce_group::all_reduce_group(const std::vector<std::string> &names)
    : _og(new_ranked_order_group(names.size()))
{
    int idx = 0;
    for (const auto &name : names) { _ranks[name] = idx++; }
}

all_reduce_group::~all_reduce_group()
{
    wait();
    del_order_group(_og);
}

void all_reduce_group::start(const std::string &name, Task task)
{
    const int rank = _ranks.at(name);
    order_group_do_rank(_og, rank, new CallbackWrapper(task));
}

void all_reduce_group::wait() { order_group_wait(_og); }

}  // namespace tensorflow
}  // namespace kungfu
