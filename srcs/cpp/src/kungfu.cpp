#include <kungfu.h>
#include <libkungfu-comm.h>

int KungfuInit(KungFu_AllReduceAlgo algo)
{
    int err = GoKungfuInit(algo);
    if (err) {
        fprintf(stderr, "%s failed\n", __func__);
        return err;
    }
    return 0;
}

int KungfuFinalize() { return GoKungfuFinalize(); }

int KungfuRequest(int destRank, 
                 void *model, int count,
                 KungFu_Datatype dtype, DoneCallback done)
{
    return GoKungfuRequest(destRank, (void *) model, GoInt(count), GoInt(dtype), new CallbackWrapper(done));
}

int KungfuRequest(int destRank, 
                 void *model, int count,
                 KungFu_Datatype dtype)
{
    return GoKungfuRequest(destRank, (void *) model, GoInt(count), GoInt(dtype), nullptr);
}


int KungfuUpdateModelStore(const char *name, const void *model, int count, KungFu_Datatype dtype, DoneCallback done)  {
    return GoKungfuUpdateModelStore((char *) name, (void *) model, GoInt(count), GoInt(dtype), new CallbackWrapper(done));
}

int KungfuReduce(const void *sendbuf, void *recvbuf, int count,
                 KungFu_Datatype dtype, KungFu_Op op, const char *name,
                 DoneCallback done)
{
    return GoKungfuReduce((void *)sendbuf, recvbuf, GoInt(count), GoInt(dtype),
                          GoInt(op), (char *)name, new CallbackWrapper(done));
}

int KungfuBroadcast(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, const char *name)
{
    return GoKungfuBroadcast((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), (char *)name, nullptr);
}

int KungfuBroadcast(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, const char *name, DoneCallback done)
{
    return GoKungfuBroadcast((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), (char *)name,
                             new CallbackWrapper(done));
}

int KungfuAllReduce(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, KungFu_Op op, const char *name)
{
    return GoKungfuAllReduce((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), GoInt(op), (char *)name, nullptr);
}

int KungfuAllReduce(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, KungFu_Op op, const char *name,
                    DoneCallback done)
{
    return GoKungfuAllReduce((void *)sendbuf, recvbuf, GoInt(count),
                             GoInt(dtype), GoInt(op), (char *)name,
                             new CallbackWrapper(done));
}

KungFu_AllReduceAlgo KungfuGetAlgoFromEnv() { return GoKungfuGetAlgoFromEnv(); }

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
    : _algo(KungfuGetAlgoFromEnv()), _global_step(0), _n_grads(0)
{
    KungfuInit(_algo);
}

kungfu_world::~kungfu_world() { KungfuFinalize(); }

int kungfu_world::Rank() const { return GoKungfuRank(); }

int kungfu_world::ClusterSize() const { return GoKungfuClusterSize(); }
