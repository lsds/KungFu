#pragma once
#include <functional>

#include <kungfu.h>

extern int KungfuInit(KungFu_AllReduceAlgo algo);

extern int KungfuFinalize();

typedef std::function<void()> DoneCallback;

extern int KungfuNegotiateAsync(const void *sendbuf, void *recvbuf, int count,
                                KungFu_Datatype datatype, KungFu_Op op,
                                const char *name, DoneCallback done);
