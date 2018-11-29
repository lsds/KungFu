#pragma once
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int Kungfu_Init();

extern int Kungfu_Finalize();

extern int Kungfu_Negotiate(const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, const char *name);

#ifdef __cplusplus
}
#endif
