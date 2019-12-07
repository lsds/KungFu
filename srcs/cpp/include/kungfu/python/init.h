#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/gpu_collective.hpp>

extern "C" {
extern void kungfu_python_init();
extern void kungfu_python_init_gpu();

// helpers APIs to access kungfu without tensorflow operators
// FIXME: don't mix with tensorflow binding
extern int kungfu_rank();          // get current rank
extern int kungfu_cluster_size();  // get current size
extern void kungfu_barrier();
}

extern std::unique_ptr<kungfu_world> _kungfu_world;

namespace kungfu
{
// order_group wraps order_group_t
class order_group
{
    order_group_t *og_;
    std::map<std::string, int> ranks_;

  public:
    using Task = DoneCallback;

    order_group(const std::vector<std::string> &names);

    ~order_group();

    void Start(const std::string &name, const Task &task);

    void Wait();
};

namespace tensorflow
{

struct cpu;
struct gpu;

template <class device> class world;

template <> class world<gpu>
{
    std::unique_ptr<gpu_collective> _gpu_collective;

    std::unique_ptr<order_group> _gpu_all_reduce_group;

  public:
    world();

    ~world();

    void StartGroup(const std::vector<std::string> &name);

    int AllReduce(DoneCallback ready, const void *sendbuf, void *recvbuf,
                  int count, KungFu_Datatype dtype, KungFu_Op op,
                  const char *name, DoneCallback done);
};

extern std::unique_ptr<world<gpu>> _world_gpu;

}  // namespace tensorflow
}  // namespace kungfu
