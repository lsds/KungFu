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
    _gpu_collective.reset(new_gpu_collective(*_kungfu_world));
}

world<gpu>::~world() {}

void world<gpu>::StartGroup(const std::vector<std::string> &names)
{
    _gpu_all_reduce_group.reset(new all_reduce_group(names));
}

int world<gpu>::AllReduce(DoneCallback ready, const void *sendbuf,
                          void *recvbuf, int count, KungFu_Datatype dtype,
                          KungFu_Op op, const char *name, DoneCallback done)
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
