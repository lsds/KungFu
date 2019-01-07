#include <string>

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

int KungfuNegotiateAsync(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, KungFu_Op op, const char *name,
                         DoneCallback done)
{
    return GoKungfuNegotiateAsync((void *)sendbuf, recvbuf, GoInt(count),
                                  GoInt(dtype), GoInt(op), (char *)name,
                                  new CallbackWrapper(done));
}

KungFu_AllReduceAlgo KungfuParseAlgoName(const char *name)
{
    return GoKungfuParseAlgoName((char *)name);
}

static std::string safe_getenv(const char *name)
{
    const char *ptr = std::getenv(name);
    if (ptr) { return std::string(ptr); }
    return "";
}

static KungFu_AllReduceAlgo get_algo_from_env()
{
    const auto name = safe_getenv("KUNGFU_ALLREDUCE_ALGO");
    return KungfuParseAlgoName(name.c_str());
}

kungfu_world::kungfu_world()
    : _algo(get_algo_from_env()), _global_step(0), _gradient_count(0)
{
    KungfuInit(_algo);
}

kungfu_world::~kungfu_world() { KungfuFinalize(); }
