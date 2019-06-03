#include <kungfu.h>
#include <libkungfu-comm.h>

order_group_t *new_ranked_order_group(int n_names)
{
    order_group_t *og = new order_group_t;
    GoNewOrderGroup(GoInt(n_names), og);
    return og;
}

void del_order_group(order_group_t *og)
{
    GoFreeOrderGroup(og);
    delete og;
}

void order_group_do_rank(order_group_t *og, int rank, callback_t *task)
{
    GoOrderGroupDoRank(og, rank, task);
}

void order_group_wait(order_group_t *og) { GoOrderGroupWait(og); }

kungfu_world::kungfu_world()
{
    int algo = GoKungfuGetAlgoFromEnv();
    int err  = GoKungfuInit(algo);
    if (err) {
        fprintf(stderr, "%s failed\n", "GoKungfuInit");
        exit(1);
    }
}

kungfu_world::~kungfu_world() { GoKungfuFinalize(); }

int kungfu_world::Rank() const { return GoKungfuRank(); }

int kungfu_world::ClusterSize() const { return GoKungfuClusterSize(); }

int kungfu_world::Request(int destRank, void *model, int count,
                          KungFu_Datatype dtype, DoneCallback done)
{
    return GoKungfuRequest(destRank, (void *)model, GoInt(count), GoInt(dtype),
                           new CallbackWrapper(done));
}

int kungfu_world::Request(int destRank, void *model, int count,
                          KungFu_Datatype dtype)
{
    return GoKungfuRequest(destRank, (void *)model, GoInt(count), GoInt(dtype),
                           nullptr);
}

int kungfu_world::UpdateModelStore(const char *name, const void *model,
                                   int count, KungFu_Datatype dtype,
                                   DoneCallback done)
{
    return GoKungfuUpdateModelStore((char *)name, (void *)model, GoInt(count),
                                    GoInt(dtype), new CallbackWrapper(done));
}

int kungfu_world::AllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name)
{
    return GoKungfuAllReduce((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), GoInt(op), (char *)name, nullptr);
}

int kungfu_world::AllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name, DoneCallback done)
{
    return GoKungfuAllReduce((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), GoInt(op), (char *)name,
                             new CallbackWrapper(done));
}

int kungfu_world::Broadcast(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, const char *name)
{
    return GoKungfuBroadcast((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), (char *)name, nullptr);
}

int kungfu_world::Broadcast(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, const char *name,
                            DoneCallback done)
{
    return GoKungfuBroadcast((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), (char *)name,
                             new CallbackWrapper(done));
}
