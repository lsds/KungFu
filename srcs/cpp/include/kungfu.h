#pragma once
#include <stddef.h>
#include <stdint.h>

#include <kungfu/dtype.h>
#include <kungfu/op.h>
#include <kungfu/strategy.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CallbackWrapper callback_t;

typedef struct order_group_s order_group_t;

extern order_group_t *new_ranked_order_group(int n_names);
extern void del_order_group(order_group_t *);
extern void order_group_do_rank(order_group_t *, int rank, callback_t *task);
extern void order_group_wait(order_group_t *);

#ifdef __cplusplus
}

#include <functional>
#include <kungfu_types.hpp>

using DoneCallback = std::function<void()>;

using TransformFunc = std::function<void(
    const void *input, int input_count, KungFu_Datatype input_dtype,
    void *output, int output_count, KungFu_Datatype output_dtype)>;

class kungfu_world
{
  public:
    kungfu_world();

    ~kungfu_world();

    // metadata APIs
    int Rank() const;
    int ClusterSize() const;

    // local API
    int Save(const char *name, const void *buf, int count,
             KungFu_Datatype dtype);
    int Save(const char *name, const void *buf, int count,
             KungFu_Datatype dtype, const DoneCallback &done);

    int Save(const char *version, const char *name, const void *buf, int count,
             KungFu_Datatype dtype);
    int Save(const char *version, const char *name, const void *buf, int count,
             KungFu_Datatype dtype, const DoneCallback &done);

    // p2p APIs
    int Request(int destRank, const char *name, void *buf, int count,
                KungFu_Datatype dtype);
    int Request(int destRank, const char *name, void *buf, int count,
                KungFu_Datatype dtype, const DoneCallback &done);

    int Request(int rank, const char *version, const char *name, void *buf,
                int count, KungFu_Datatype dtype);
    int Request(int rank, const char *version, const char *name, void *buf,
                int count, KungFu_Datatype dtype, const DoneCallback &done);

    // collective APIs
    int Barrier();
    int Barrier(const DoneCallback &done);

    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Reduce.3.php
    int Reduce(const void *sendbuf, void *recvbuf, int count,
               KungFu_Datatype dtype, KungFu_Op op, const char *name,
               const DoneCallback &done);

    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Allreduce.3.php
    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name);
    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name,
                  const DoneCallback &done);

    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Bcast.3.php
    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name);
    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name,
                  const DoneCallback &done);

    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Gather.3.php
    int Gather(const void *sendbuf, int send_count, KungFu_Datatype send_dtype,
               void *recvbuf, int recv_count, KungFu_Datatype recv_dtype,
               const char *name);
    int Gather(const void *sendbuf, int send_count, KungFu_Datatype send_dtype,
               void *recvbuf, int recv_count, KungFu_Datatype recv_dtype,
               const char *name, const DoneCallback &done);

    //  highlevel APIs
    int AllGatherTransform(const void *input, int input_count,
                           KungFu_Datatype input_dtype,  //
                           void *output, int output_count,
                           KungFu_Datatype output_dtype,  //
                           const char *name, const TransformFunc &f);

    template <typename T1, typename T2, typename F>
    int AllGatherTransform(const T1 *input, int input_count,  //
                           T2 *output, int output_count,      //
                           const char *name, const F &f)
    {
        return AllGatherTransform(
            input, input_count, kungfu::type_encoder::value<T1>(),  //
            output, output_count, kungfu::type_encoder::value<T2>(), name,
            [f = f](const void *input, int input_count,
                    KungFu_Datatype input_dtype, void *output, int output_count,
                    KungFu_Datatype output_dtype) {
                f(reinterpret_cast<const T1 *>(input), input_count,
                  reinterpret_cast<T2 *>(output), output_count);
            });
    }

    // monitoring APIs
    int GetPeerLatencies(float *recvbuf, int recv_count);

    // control APIs
    int ResizeCluster(const char *ckpt, int new_size, bool *keep);
};

#endif
