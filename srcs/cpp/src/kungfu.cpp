#include <kungfu.h>
#include <libkungfu-comm.h>

#include <vector>

order_group_t *new_ranked_order_group(int n_names)
{
    order_group_t *og = new order_group_t;
    GoNewOrderGroup(GoInt(n_names), og);
    return og;
}

void del_order_group(order_group_t *og)
{
    GoFreeOrderGroup(og);
    delete og;
}

void order_group_do_rank(order_group_t *og, int rank, callback_t *task)
{
    GoOrderGroupDoRank(og, rank, task);
}

void order_group_wait(order_group_t *og, int32_t *arrive_order)
{
    GoOrderGroupWait(og, arrive_order);
}

kungfu_world::kungfu_world()
{
    const int err = GoKungfuInit();
    if (err) {
        fprintf(stderr, "%s failed\n", "GoKungfuInit");
        exit(1);
    }
}

kungfu_world::~kungfu_world() { GoKungfuFinalize(); }

uint64_t kungfu_world::Uid() const { return GoKungfuUID(); }

int kungfu_world::Rank() const { return GoKungfuRank(); }

int kungfu_world::LocalRank() const { return GoKungfuLocalRank(); }

int kungfu_world::ClusterSize() const { return GoKungfuClusterSize(); }

int kungfu_world::Save(const char *name, const void *buf, int count,
                       KungFu_Datatype dtype)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), dtype, nullptr);
}

int kungfu_world::Save(const char *name, const void *buf, int count,
                       KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), dtype, new CallbackWrapper(done));
}

int kungfu_world::Save(const char *version, const char *name, const void *buf,
                       int count, KungFu_Datatype dtype)
{
    return GoKungfuSaveVersion(
        const_cast<char *>(version), const_cast<char *>(name),
        const_cast<void *>(buf), GoInt(count), dtype, nullptr);
}

int kungfu_world::Save(const char *version, const char *name, const void *buf,
                       int count, KungFu_Datatype dtype,
                       const DoneCallback &done)
{
    return GoKungfuSaveVersion(const_cast<char *>(version),
                               const_cast<char *>(name),
                               const_cast<void *>(buf), GoInt(count), dtype,
                               new CallbackWrapper(done));
}

int kungfu_world::Barrier() { return GoKungfuBarrier(nullptr); }

int kungfu_world::Barrier(const DoneCallback &done)
{
    return GoKungfuBarrier(new CallbackWrapper(done));
}

int kungfu_world::Consensus(const void *buf, int count, KungFu_Datatype dtype,
                            bool *ok, const char *name)
{
    return GoKungfuConsensus(const_cast<void *>(buf), GoInt(count), dtype,
                             reinterpret_cast<char *>(ok),
                             const_cast<char *>(name), nullptr);
}

int kungfu_world::Consensus(const void *buf, int count, KungFu_Datatype dtype,
                            bool *ok, const char *name,
                            const DoneCallback &done)
{
    return GoKungfuConsensus(const_cast<void *>(buf), GoInt(count), dtype,
                             reinterpret_cast<char *>(ok),
                             const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int kungfu_world::Request(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, nullptr);
}

int kungfu_world::Request(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, new CallbackWrapper(done));
}

int kungfu_world::SpotnikRequest(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype, int32_t *succeeded, const DoneCallback &done)
{
    return GoSpotnikRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, succeeded, new CallbackWrapper(done));
}

int kungfu_world::Request(int rank, const char *version, const char *name,
                          void *buf, int count, KungFu_Datatype dtype)
{
    return GoKungfuRequestVersion(rank, const_cast<char *>(version),
                                  const_cast<char *>(name), buf, GoInt(count),
                                  dtype, nullptr);
}

int kungfu_world::Request(int rank, const char *version, const char *name,
                          void *buf, int count, KungFu_Datatype dtype,
                          const DoneCallback &done)
{
    return GoKungfuRequestVersion(rank, const_cast<char *>(version),
                                  const_cast<char *>(name), buf, GoInt(count),
                                  dtype, new CallbackWrapper(done));
}

int kungfu_world::Reduce(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, KungFu_Op op, const char *name,
                         const DoneCallback &done)
{
    return GoKungfuReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                          dtype, op, const_cast<char *>(name),
                          new CallbackWrapper(done));
}

int kungfu_world::AllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name)
{
    return GoKungfuAllReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             dtype, op, const_cast<char *>(name), nullptr);
}

int kungfu_world::AllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name, const DoneCallback &done)
{
    return GoKungfuAllReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             dtype, op, const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int kungfu_world::SpotnikAllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, int32_t *succeeded, KungFu_Op op,
                            const char *name, const DoneCallback &done)
{
    return GoSpotnikAllReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             dtype, succeeded, op, const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int kungfu_world::Broadcast(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, const char *name)
{
    return GoKungfuBroadcast(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             dtype, const_cast<char *>(name), nullptr);
}

int kungfu_world::Broadcast(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, const char *name,
                            const DoneCallback &done)
{
    return GoKungfuBroadcast(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             dtype, const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int kungfu_world::Gather(const void *sendbuf, int send_count,
                         KungFu_Datatype send_dtype, void *recvbuf,
                         int recv_count, KungFu_Datatype recv_dtype,
                         const char *name)
{
    return GoKungfuGather(const_cast<void *>(sendbuf), GoInt(send_count),
                          send_dtype, recvbuf, GoInt(recv_count), recv_dtype,
                          const_cast<char *>(name), nullptr);
}

int kungfu_world::Gather(const void *sendbuf, int send_count,
                         KungFu_Datatype send_dtype, void *recvbuf,
                         int recv_count, KungFu_Datatype recv_dtype,
                         const char *name, const DoneCallback &done)
{
    return GoKungfuGather(const_cast<void *>(sendbuf), GoInt(send_count),
                          send_dtype, recvbuf, GoInt(recv_count), recv_dtype,
                          const_cast<char *>(name), new CallbackWrapper(done));
}

int kungfu_world::AllGatherTransform(const void *input, int input_count,
                                     KungFu_Datatype input_dtype,  //
                                     void *output, int output_count,
                                     KungFu_Datatype output_dtype,  //
                                     const char *name, const TransformFunc &f)
{
    const bool is_root = Rank() == 0;  // FIXME: make sure 0 is root
    if (is_root) {
        const int total_count = ClusterSize() * input_count;
        std::vector<char> all_input(total_count *
                                    kungfu_type_size(input_dtype));
        Gather(input, input_count, input_dtype, all_input.data(), total_count,
               input_dtype, name);
        f(all_input.data(), total_count, input_dtype,  //
          output, output_count, output_dtype);
    } else {
        Gather(input, input_count, input_dtype, nullptr, 0, input_dtype, name);
    }
    return Broadcast(output, output, output_count, output_dtype, name);
}

// monitoring APIs
int kungfu_world::GetPeerLatencies(float *recvbuf, int recv_count)
{
    return GoKungfuGetPeerLatencies(recvbuf, recv_count, KungFu_FLOAT);
}

// control APIs
int kungfu_world::ResizeClusterFromURL(bool *changed, bool *keep)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuResizeClusterFromURL(reinterpret_cast<char *>(changed),
                                        reinterpret_cast<char *>(keep));
}

int kungfu_world::ProposeNewSize(int new_size)
{
    return GoKungfuProposeNewSize(GoInt(new_size));
}
