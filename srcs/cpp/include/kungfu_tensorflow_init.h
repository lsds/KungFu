#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu_gpu_collective.hpp>

extern "C" {
extern void kungfu_tensorflow_init();
}

extern std::unique_ptr<kungfu_world> _kungfu_bootstrap;

extern kungfu_world *_kungfu_world;  // FIXME: remove

namespace kungfu
{
namespace tensorflow
{
struct cpu;
struct gpu;

class all_reduce_group
{
    order_group_t *_og;
    std::map<std::string, int> _ranks;

  public:
    using Task = DoneCallback;

    all_reduce_group(const std::vector<std::string> &names);

    ~all_reduce_group();

    void start(const std::string &name, Task task);

    void wait();
};

class world
{
    std::unique_ptr<gpu_collective> _gpu_collective;

    std::unique_ptr<all_reduce_group> _gpu_all_reduce_group;

  public:
    world();

    ~world();

    void init_gpu_collective();

    int32_t AdvanceGlobalStep()
    {
        return _kungfu_bootstrap->AdvanceGlobalStep();
    }

    void SetNumGradients(int32_t n_grads)
    {
        _kungfu_bootstrap->SetNumGradients(n_grads);
    }

    void StartGpuGroup(const std::vector<std::string> &name);

    int AllReduceGpu(DoneCallback ready, const void *sendbuf, void *recvbuf,
                     int count, KungFu_Datatype dtype, KungFu_Op op,
                     const char *name, DoneCallback done);
};

extern std::unique_ptr<world> _world;

}  // namespace tensorflow
}  // namespace kungfu
