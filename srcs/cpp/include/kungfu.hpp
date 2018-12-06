#pragma once
#include <functional>

#include <mpi.h>

extern int KungfuInit();

extern int KungfuFinalize();

typedef std::function<void()> DoneCallback;

extern int KungfuNegotiateAsync(const void *sendbuf, void *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op,
                                const char *name, DoneCallback done);
