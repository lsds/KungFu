#include <cstdio>
#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_init.h>

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_tensorflow_init()
{
    _kungfu_world.reset(new kungfu_world);
    kungfu::tensorflow::_world.reset(new kungfu::tensorflow::world);

    bool have_gpu = true;  // FIXME: auto detect
    if (have_gpu) { kungfu::tensorflow::_world->init_gpu_collective(); }
}

namespace kungfu
{
namespace tensorflow
{
std::unique_ptr<world> _world;

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

world::world() {}

world::~world() {}

void world::init_gpu_collective()
{
    _gpu_collective.reset(new_gpu_collective(*_kungfu_world));
}

void world::StartGpuGroup(const std::vector<std::string> &names)
{
    _gpu_all_reduce_group.reset(new all_reduce_group(names));
}

int world::AllReduceGpu(DoneCallback ready, const void *sendbuf, void *recvbuf,
                        int count, KungFu_Datatype dtype, KungFu_Op op,
                        const char *name, DoneCallback done)
{
    _gpu_all_reduce_group->start(name, [=, comm = _gpu_collective.get()]() {
        ready();
        comm->all_reduce(sendbuf, recvbuf, count, dtype);
        done();
    });
    return 0;
}

}  // namespace tensorflow
}  // namespace kungfu
