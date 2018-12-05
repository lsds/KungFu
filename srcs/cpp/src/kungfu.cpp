#include "cgo_helpers.hpp"
#include <kungfu.h>

int KungfuInit()
{
    int err = GoKungfuInit();
    if (err) {
        fprintf(stderr, "%s failed\n", __func__);
        return err;
    }
    return 0;
}

int KungfuFinalize() { return GoKungfuFinalize(); }

int KungfuNegotiate(const void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, const char *name)
{
    auto gs_send = toGoSlice(sendbuf, count, datatype);
    auto gs_recv = toGoSlice(recvbuf, count, datatype);
    auto go_name = toGoString(name);
    return GoKungfuNegotiate(gs_send, gs_recv, GoInt(count), GoInt(datatype),
                             GoInt(op), go_name);
}
