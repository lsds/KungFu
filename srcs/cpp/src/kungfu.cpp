#include "cgo_helpers.hpp"
#include <kungfu.hpp>

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
