#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int KungFu_Datatype;

// extern const KungFu_Datatype KungFu_INT8;
// extern const KungFu_Datatype KungFu_UINT8;
// extern const KungFu_Datatype KungFu_INT16;
// extern const KungFu_Datatype KungFu_UINT16;
extern const KungFu_Datatype KungFu_INT32;
// extern const KungFu_Datatype KungFu_UINT32;
// extern const KungFu_Datatype KungFu_INT64;
// extern const KungFu_Datatype KungFu_UINT64;
extern const KungFu_Datatype KungFu_FLOAT;
extern const KungFu_Datatype KungFu_DOUBLE;
// extern const KungFu_Datatype KungFu_LONG_DOUBLE;

extern uint32_t kungfu_type_size(KungFu_Datatype);

typedef int KungFu_Op;

extern const KungFu_Op KungFu_MAX;
extern const KungFu_Op KungFu_MIN;
extern const KungFu_Op KungFu_SUM;

typedef int KungFu_AllReduceAlgo;

extern const KungFu_AllReduceAlgo KungFu_StarAllReduce;
extern const KungFu_AllReduceAlgo KungFu_RingAllReduce;
extern const KungFu_AllReduceAlgo KungFu_CliqueAllReduce;
extern const KungFu_AllReduceAlgo KungFu_TreeAllReduce;
// extern KungFu_AllReduceAlgo KungFu_DynamicAllReduce;

extern int KungfuInit(KungFu_AllReduceAlgo algo);

extern int KungfuFinalize();

extern KungFu_AllReduceAlgo KungfuGetAlgoFromEnv();

#ifdef __cplusplus
}

#include <functional>
typedef std::function<void()> DoneCallback;

extern int KungfuReduce(const void *sendbuf, void *recvbuf, int count,
                        KungFu_Datatype dtype, KungFu_Op op, const char *name,
                        DoneCallback done);

extern int KungfuAllReduce(const void *sendbuf, void *recvbuf, int count,
                           KungFu_Datatype dtype, KungFu_Op op,
                           const char *name);

extern int KungfuAllReduce(const void *sendbuf, void *recvbuf, int count,
                           KungFu_Datatype dtype, KungFu_Op op,
                           const char *name, DoneCallback done);

class kungfu_world
{
    KungFu_AllReduceAlgo _algo;
    int32_t _global_step;
    int32_t _n_grads;

  public:
    kungfu_world();

    ~kungfu_world();

    int32_t AdvanceGlobalStep() { return ++_global_step; }

    void SetNumGradients(int32_t n_grads) { _n_grads = n_grads; }

    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name,
                  DoneCallback done)
    {
        return KungfuAllReduce(sendbuf, recvbuf, count, dtype, op, name, done);
    }
};

#endif
