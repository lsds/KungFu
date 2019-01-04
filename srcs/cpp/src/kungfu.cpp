#include <map>
#include <string>

#include "cgo_helpers.hpp"
#include <kungfu.h>

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
                         KungFu_Datatype datatype, KungFu_Op op,
                         const char *name, DoneCallback done)
{
    auto gs_send = toGoSlice(sendbuf, count, datatype);
    auto gs_recv = toGoSlice(recvbuf, count, datatype);
    auto go_name = toGoString(name);
    return GoKungfuNegotiateAsync(gs_send, gs_recv, GoInt(count),
                                  GoInt(datatype), GoInt(op), go_name,
                                  new CallbackWrapper(done));
}

static const std::map<std::string, KungFu_AllReduceAlgo> _kungfu_algo_names({
    {"SIMPLE", KungFu_SimpleAllReduce},
    {"RING", KungFu_RingAllReduce},
    {"CLIQUE", KungFu_FullSymmetricAllReduce},
    {"TREE", KungFu_TreeAllReduce},
});

KungFu_AllReduceAlgo kungfu_parse_algo_name(const char *name)
{
    if (_kungfu_algo_names.count(name) > 0) {
        return _kungfu_algo_names.at(name);
    }
    return KungFu_TreeAllReduce;
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
    return kungfu_parse_algo_name(name.c_str());
}

kungfu_world::kungfu_world() : _algo(get_algo_from_env()), _global_step(0)
{
    KungfuInit(_algo);
}

kungfu_world::~kungfu_world() { KungfuFinalize(); }
