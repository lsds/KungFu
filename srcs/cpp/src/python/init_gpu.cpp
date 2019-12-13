#include <kungfu.h>
#include <kungfu/python/init.h>

void kungfu_python_init_gpu()
{
    kungfu::tensorflow::_world_gpu.reset(
        new kungfu::tensorflow::world<kungfu::tensorflow::gpu>);
}

void kungfu_python_finialize_gpu()
{
    kungfu::tensorflow::_world_gpu.reset(nullptr);
}

namespace kungfu
{
namespace tensorflow
{
std::unique_ptr<world<gpu>> _world_gpu;

world<gpu>::world()
{
    _gpu_collective.reset(new_gpu_collective(*_kungfu_world));
}

int world<gpu>::ScheduledAllReduce(DoneCallback ready, const void *sendbuf,
                                   void *recvbuf, int count,
                                   KungFu_Datatype dtype, KungFu_Op op,
                                   const char *name, DoneCallback done)
{
    kungfu::_nccl_order_group->Start(name, [=, comm = _gpu_collective.get()]() {
        ready();
        comm->all_reduce(sendbuf, recvbuf, count, dtype);
        done();
    });
    return 0;
}

int world<gpu>::AllReduce(const void *sendbuf, void *recvbuf, int count,
                          KungFu_Datatype dtype, KungFu_Op op, const char *name,
                          DoneCallback done)
{
    _gpu_collective.get()->all_reduce(sendbuf, recvbuf, count, dtype);
    done();
    return 0;
}
}  // namespace tensorflow
}  // namespace kungfu
