#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int KungFu_Datatype;

// extern const KungFu_Datatype KungFu_INT8;
extern const KungFu_Datatype KungFu_UINT8;
// extern const KungFu_Datatype KungFu_INT16;
// extern const KungFu_Datatype KungFu_UINT16;
extern const KungFu_Datatype KungFu_INT32;
// extern const KungFu_Datatype KungFu_UINT32;
extern const KungFu_Datatype KungFu_INT64;
// extern const KungFu_Datatype KungFu_UINT64;
extern const KungFu_Datatype KungFu_FLOAT16;
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

typedef struct CallbackWrapper callback_t;

typedef struct order_group_s order_group_t;

extern order_group_t *new_ranked_order_group(int n_names);
extern void del_order_group(order_group_t *);
extern void order_group_do_rank(order_group_t *, int rank, callback_t *task);
extern void order_group_wait(order_group_t *);

#ifdef __cplusplus
}

#include <functional>
typedef std::function<void()> DoneCallback;
typedef std::function<void(void *,int)> DataCallback;

extern int KungfuReduce(const void *sendbuf, void *recvbuf, int count,
                        KungFu_Datatype dtype, KungFu_Op op, const char *name,
                        DoneCallback done);

// broadcast the data from the current root in the first pair of graphs.
extern int KungfuBroadcast(const void *sendbuf, void *recvbuf, int count,
                           KungFu_Datatype dtype, const char *name);

extern int KungfuBroadcast(const void *sendbuf, void *recvbuf, int count,
                           KungFu_Datatype dtype, const char *name,
                           DoneCallback done);

extern int KungfuSendTo(int32_t rank, const void *sendbuf, int count,
                        KungFu_Datatype dtype, const char *name,
                        DoneCallback done);
extern int KungfuSendTo(int32_t rank, const void *sendbuf, int count,
                        KungFu_Datatype dtype, const char *name);

extern int KungfuRequestVar(int32_t rank, const char *name, int count, KungFu_Datatype dtype, void *recvbuf);

extern int KungfuRegisterDataCallback(const char *name, DataCallback handle);

extern int KungfuUnregisterDataCallback(const char *name);

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

    int Rank() const;

    int ClusterSize() const;

    int32_t AdvanceGlobalStep() { return ++_global_step; }

    int32_t GetGlobalStep() { return _global_step; }

    void SetNumGradients(int32_t n_grads) { _n_grads = n_grads; }

    int SendTo(int32_t rank, const void *sendbuf, int count,
               KungFu_Datatype dtype, const char *name, DoneCallback done)
    {
        return KungfuSendTo(rank, sendbuf, count, dtype, name, done);
    }

    int SendTo(int32_t rank, const void *sendbuf, int count,
               KungFu_Datatype dtype, const char *name)
    {
        return KungfuSendTo(rank, sendbuf, count, dtype, name);
    }

    int RequestVar(int32_t rank, const char *name, int count, KungFu_Datatype dtype, void *recvbuf)
    {
        return KungfuRequestVar(rank, name, count, dtype, recvbuf);
    }

    int RegisterDataCallback(const char *name, DataCallback handle)
    {
        return KungfuRegisterDataCallback(name, handle);
    }

    int UnregisterDataCallback(const char *name)
    {
        return KungfuUnregisterDataCallback(name);
    }

    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name)
    {
        return KungfuAllReduce(sendbuf, recvbuf, count, dtype, op, name);
    }

    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name,
                  DoneCallback done)
    {
        return KungfuAllReduce(sendbuf, recvbuf, count, dtype, op, name, done);
    }

    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name)
    {
        return KungfuBroadcast(sendbuf, recvbuf, count, dtype, name);
    }

    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name, DoneCallback done)
    {
        return KungfuBroadcast(sendbuf, recvbuf, count, dtype, name, done);
    }
};

#endif
