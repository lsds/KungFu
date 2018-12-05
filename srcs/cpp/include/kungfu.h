#pragma once
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int KungfuInit();

extern int KungfuFinalize();

extern int KungfuNegotiate(const void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, const char *name);

#ifdef __cplusplus
}
#endif
