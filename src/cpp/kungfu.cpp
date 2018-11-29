#include "cgo-helpers.hpp"
#include <kungfu.h>

int Kungfu_Init()
{
    int err = Go_Kungfu_Init();
    if (err) {
        fprintf(stderr, "%s failed\n", __func__);
        return err;
    }
    return 0;
}

int Kungfu_Finalize() { return Go_Kungfu_Finalize(); }

int Kungfu_Negotiate(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op op, const char *name)
{
    auto gs_send = to_go_slice(sendbuf, count, datatype);
    auto gs_recv = to_go_slice(recvbuf, count, datatype);
    auto go_name = to_go_string(name);
    return Go_Kungfu_Negotiate(gs_send, gs_recv, GoInt(count), GoInt(datatype),
                               GoInt(op), go_name);
}
