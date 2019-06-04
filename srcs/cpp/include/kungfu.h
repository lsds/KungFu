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

class kungfu_world
{
  public:
    kungfu_world();

    ~kungfu_world();

    int Rank() const;

    int ClusterSize() const;

    int UpdateModelStore(const char *model_version_name, const void *model,
                         int count, KungFu_Datatype dtype,
                         const DoneCallback &done);

    // p2p APIs
    int Request(int destRank, void *model, int count, KungFu_Datatype dtype);
    int Request(int destRank, void *model, int count, KungFu_Datatype dtype,
                const DoneCallback &done);

    // collective APIs
    int Reduce(const void *sendbuf, void *recvbuf, int count,
               KungFu_Datatype dtype, KungFu_Op op, const char *name,
               const DoneCallback &done);

    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name);
    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name,
                  const DoneCallback &done);

    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name);
    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name,
                  const DoneCallback &done);
};

#endif
