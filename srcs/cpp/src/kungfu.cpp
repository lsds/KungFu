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

void order_group_wait(order_group_t *og) { GoOrderGroupWait(og); }

kungfu_world::kungfu_world()
{
    int algo = GoKungfuGetAlgoFromEnv();
    int err  = GoKungfuInit(algo);
    if (err) {
        fprintf(stderr, "%s failed\n", "GoKungfuInit");
        exit(1);
    }
}

kungfu_world::~kungfu_world() { GoKungfuFinalize(); }

int kungfu_world::Rank() const { return GoKungfuRank(); }

int kungfu_world::ClusterSize() const { return GoKungfuClusterSize(); }

int kungfu_world::Save(const char *name, const void *buf, int count,
                       KungFu_Datatype dtype)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), GoInt(dtype), nullptr);
}

int kungfu_world::Save(const char *name, const void *buf, int count,
                       KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), GoInt(dtype), new CallbackWrapper(done));
}

int kungfu_world::Barrier(const DoneCallback &done)
{
    return GoKungfuBarrier(new CallbackWrapper(done));
}

int kungfu_world::Request(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), GoInt(dtype), nullptr);
}

int kungfu_world::Request(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), GoInt(dtype),
                           new CallbackWrapper(done));
}

int kungfu_world::Reduce(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, KungFu_Op op, const char *name,
                         const DoneCallback &done)
{
    return GoKungfuReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                          GoInt(dtype), GoInt(op), const_cast<char *>(name),
                          new CallbackWrapper(done));
}

int kungfu_world::AllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name)
{
    return GoKungfuAllReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             GoInt(dtype), GoInt(op), const_cast<char *>(name),
                             nullptr);
}

int kungfu_world::AllReduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name, const DoneCallback &done)
{
    return GoKungfuAllReduce(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             GoInt(dtype), GoInt(op), const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int kungfu_world::Broadcast(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, const char *name)
{
    return GoKungfuBroadcast(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             GoInt(dtype), const_cast<char *>(name), nullptr);
}

int kungfu_world::Broadcast(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, const char *name,
                            const DoneCallback &done)
{
    return GoKungfuBroadcast(const_cast<void *>(sendbuf), recvbuf, GoInt(count),
                             GoInt(dtype), const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int kungfu_world::Gather(const void *sendbuf, int send_count,
                         KungFu_Datatype send_dtype, void *recvbuf,
                         int recv_count, KungFu_Datatype recv_dtype,
                         const char *name)
{
    return GoKungfuGather(const_cast<void *>(sendbuf), GoInt(send_count),
                          GoInt(send_dtype), recvbuf, GoInt(recv_count),
                          GoInt(recv_dtype), const_cast<char *>(name), nullptr);
}

int kungfu_world::Gather(const void *sendbuf, int send_count,
                         KungFu_Datatype send_dtype, void *recvbuf,
                         int recv_count, KungFu_Datatype recv_dtype,
                         const char *name, const DoneCallback &done)
{
    return GoKungfuGather(const_cast<void *>(sendbuf), GoInt(send_count),
                          GoInt(send_dtype), recvbuf, GoInt(recv_count),
                          GoInt(recv_dtype), const_cast<char *>(name),
                          new CallbackWrapper(done));
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
int kungfu_world::ProposeUpdate(const char *token, bool *result)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuProposeUpdate(const_cast<char *>(token),
                                 reinterpret_cast<char *>(result));
}

int kungfu_world::UpdateCluster(const char *token, bool *exist)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuUpdateCluster(const_cast<char *>(token),
                                 reinterpret_cast<char *>(exist));
}
