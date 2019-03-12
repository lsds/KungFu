#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_init.h>

void kungfu_tensorflow_init_gpu()
{
    kungfu::tensorflow::_world_gpu.reset(
        new kungfu::tensorflow::world<kungfu::tensorflow::gpu>);
}

namespace kungfu
{
namespace tensorflow
{
std::unique_ptr<world<gpu>> _world_gpu;

world<gpu>::world()
{
    _local_nccl_comm.reset(new_local_nccl_comm(*_kungfu_world));
}

world<gpu>::~world() {}

void world<gpu>::StartGroup(const std::vector<std::string> &names)
{
    _local_reduce_group.reset(new order_group(names));
    _local_bcast_group.reset(new order_group(names));
}

int world<gpu>::AllReduce(DoneCallback ready, const void *sendbuf,
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
