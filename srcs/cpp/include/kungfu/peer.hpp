#pragma once
#include <functional>
#include <kungfu/dtype.hpp>

namespace kungfu
{
using DoneCallback = std::function<void()>;

using TransformFunc = std::function<void(
    const void *input, int input_count, KungFu_Datatype input_dtype,
    void *output, int output_count, KungFu_Datatype output_dtype)>;

class Peer
{
  public:
    Peer();                    // init from env
    Peer(int rank, int size);  // Single Machine Multi-Process
    Peer(const char *pJson);   // init from JSON

    ~Peer();

    bool Detached() const;

    // metadata APIs
    uint64_t Uid() const;

    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_rank.3.php
    int Rank() const;

    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_size.3.php
    int Size() const;

    int LocalRank() const;
    int LocalSize() const;
    int HostCount() const;

    // call Done asynchronously
    int Noop(const DoneCallback &done);

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

    // FIXME: move Session APIs to Session class in C++

    // collective APIs
    int Barrier();
    int Barrier(const DoneCallback &done);

    int Consensus(const void *buf, int count, KungFu_Datatype dtype, bool *ok,
                  const char *name);
    int Consensus(const void *buf, int count, KungFu_Datatype dtype, bool *ok,
                  const char *name, const DoneCallback &done);

    // variant of https://www.open-mpi.org/doc/v4.0/man3/MPI_Reduce.3.php
    int Reduce(const void *sendbuf, void *recvbuf, int count,
               KungFu_Datatype dtype, KungFu_Op op, const char *name,
               const DoneCallback &done);

    // variant of https://www.open-mpi.org/doc/v4.0/man3/MPI_Allreduce.3.php
    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name);
    int AllReduce(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, KungFu_Op op, const char *name,
                  const DoneCallback &done);

    int CrossAllReduce(const void *sendbuf, void *recvbuf, int count,
                       KungFu_Datatype dtype, KungFu_Op op, const char *name);
    int CrossAllReduce(const void *sendbuf, void *recvbuf, int count,
                       KungFu_Datatype dtype, KungFu_Op op, const char *name,
                       const DoneCallback &done);

    int MonitoredAllReduce(const void *sendbuf, void *recvbuf, int count,
                           KungFu_Datatype dtype, KungFu_Op op,
                           const int32_t *tree, const char *name,
                           const DoneCallback &done);

    // variant of https://www.open-mpi.org/doc/v4.0/man3/MPI_Allgather.3.php
    int AllGather(const void *sendbuf, int count, KungFu_Datatype dtype,
                  void *recvbuf, const char *name);
    int AllGather(const void *sendbuf, int count, KungFu_Datatype dtype,
                  void *recvbuf, const char *name, const DoneCallback &done);

    // variant of https://www.open-mpi.org/doc/v4.0/man3/MPI_Bcast.3.php
    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name);
    int Broadcast(const void *sendbuf, void *recvbuf, int count,
                  KungFu_Datatype dtype, const char *name,
                  const DoneCallback &done);

    int LocalBroadcast(const void *sendbuf, void *recvbuf, int count,
                       KungFu_Datatype dtype, const char *name);
    int LocalBroadcast(const void *sendbuf, void *recvbuf, int count,
                       KungFu_Datatype dtype, const char *name,
                       const DoneCallback &done);

    // variant of https://www.open-mpi.org/doc/v4.0/man3/MPI_Gather.3.php
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

    // adaptation APIs
    int SetTree(const int32_t *tree);

    // monitoring APIs
    int GetPeerLatencies(float *recvbuf, int recv_count);
    int CheckInterference();
    int GetEgressRates(float *rates);

    // control APIs
    int ResizeCluster(const uint32_t new_size, bool *changed, bool *detached);
    int ResizeClusterFromURL(bool *changed, bool *detached);

    int ProposeNewSize(int new_size);

    void CalcStats();
    void LogStats();
    void PrintStategyStats();
};
}  // namespace kungfu
