#include <cstdio>
#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_comm.hpp>
#include <kungfu_tensorflow_init.h>

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_tensorflow_init() { _kungfu_world.reset(new kungfu_world); }

namespace kungfu
{

order_group::order_group(const std::vector<std::string> &names)
    : _og(new_ranked_order_group(names.size()))
{
    int idx = 0;
    for (const auto &name : names) { _ranks[name] = idx++; }
}

order_group::~order_group()
{
    wait();
    del_order_group(_og);
}

void order_group::start(const std::string &name, Task task)
{
    const int rank = _ranks.at(name);
    order_group_do_rank(_og, rank, new CallbackWrapper(task));
}

void order_group::wait() { order_group_wait(_og); }

namespace tensorflow
{
std::unique_ptr<world<cpu>> _world_cpu;

world<cpu>::world()
{
    // _local_nccl_comm.reset(new_local_go_comm(*_kungfu_world));
    _inter_host_comm.reset(new_inter_go_comm(*_kungfu_world));
}

world<cpu>::~world() {}

void world<cpu>::StartGroup(const std::vector<std::string> &names)
{
    _local_reduce_group.reset(new order_group(names));
    _local_bcast_group.reset(new order_group(names));
}

int world<cpu>::AllReduce(DoneCallback ready, const void *sendbuf,
                          void *recvbuf, int count, KungFu_Datatype dtype,
                          KungFu_Op op, const char *name, DoneCallback done)
{
    auto _local_comm = _local_nccl_comm.get();
    auto _inter_comm = _inter_host_comm.get();

    _local_reduce_group->start(name, [=] {
        ready();
        _local_comm->reduce(sendbuf, recvbuf, count, dtype);
        _inter_comm->all_reduce(recvbuf, recvbuf, count, dtype, name, [=] {
            _local_bcast_group->start(name, [=] {
                _local_comm->bcast(recvbuf, recvbuf, count, dtype);
                done();
            });
        });
    });

    return 0;
}

}  // namespace tensorflow

}  // namespace kungfu
