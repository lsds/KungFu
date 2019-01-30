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

kungfu_world::kungfu_world()
    : _algo(KungfuGetAlgoFromEnv()), _global_step(0), _n_grads(0)
{
    KungfuInit(_algo);
}

kungfu_world::~kungfu_world() { KungfuFinalize(); }
