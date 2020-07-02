#include <vector>

#include <kungfu/session.hpp>
#include <libkungfu-comm.h>

namespace kungfu
{
int Peer::Rank() const { return GoKungfuRank(p_); }

int Peer::Size() const { return GoKungfuSize(p_); }

int Peer::LocalRank() const { return GoKungfuLocalRank(p_); }

int Peer::LocalSize() const { return GoKungfuLocalSize(p_); }

int Peer::HostCount() const { return GoKungfuHostCount(p_); }

int Peer::Barrier() { return GoKungfuBarrier(p_, nullptr); }

int Peer::Barrier(const DoneCallback &done)
{
    return GoKungfuBarrier(p_, new CallbackWrapper(done));
}

int Peer::Consensus(const void *buf, int count, KungFu_Datatype dtype, bool *ok,
                    const char *name)
{
    return GoKungfuConsensus(p_, const_cast<void *>(buf), GoInt(count), dtype,
                             reinterpret_cast<char *>(ok),
                             const_cast<char *>(name), nullptr);
}

int Peer::Consensus(const void *buf, int count, KungFu_Datatype dtype, bool *ok,
                    const char *name, const DoneCallback &done)
{
    return GoKungfuConsensus(p_, const_cast<void *>(buf), GoInt(count), dtype,
                             reinterpret_cast<char *>(ok),
                             const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int Peer::Reduce(const void *sendbuf, void *recvbuf, int count,
                 KungFu_Datatype dtype, KungFu_Op op, const char *name,
                 const DoneCallback &done)
{
    return GoKungfuReduce(p_, const_cast<void *>(sendbuf), recvbuf,
                          GoInt(count), dtype, op, const_cast<char *>(name),
                          new CallbackWrapper(done));
}

int Peer::AllReduce(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, KungFu_Op op, const char *name)
{
    return GoKungfuAllReduce(p_, const_cast<void *>(sendbuf), recvbuf,
                             GoInt(count), dtype, op, const_cast<char *>(name),
                             nullptr);
}

int Peer::AllReduce(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, KungFu_Op op, const char *name,
                    const DoneCallback &done)
{
    return GoKungfuAllReduce(p_, const_cast<void *>(sendbuf), recvbuf,
                             GoInt(count), dtype, op, const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int Peer::CrossAllReduce(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, KungFu_Op op, const char *name)
{
    return GoKungfuCrossAllReduce(p_, const_cast<void *>(sendbuf), recvbuf,
                                  GoInt(count), dtype, op,
                                  const_cast<char *>(name), nullptr);
}

int Peer::CrossAllReduce(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, KungFu_Op op, const char *name,
                         const DoneCallback &done)
{
    return GoKungfuCrossAllReduce(
        p_, const_cast<void *>(sendbuf), recvbuf, GoInt(count), dtype, op,
        const_cast<char *>(name), new CallbackWrapper(done));
}

int Peer::MonitoredAllReduce(const void *sendbuf, void *recvbuf, int count,
                             KungFu_Datatype dtype, KungFu_Op op,
                             const int32_t *tree, const char *name,
                             const DoneCallback &done)
{
    return GoKungfuMonitoredAllReduce(
        p_, const_cast<void *>(sendbuf), recvbuf, GoInt(count), dtype, op,
        const_cast<int32_t *>(tree), const_cast<char *>(name),
        new CallbackWrapper(done));
}

int Peer::AllGather(const void *sendbuf, int count, KungFu_Datatype dtype,
                    void *recvbuf, const char *name)
{
    return GoKungfuAllGather(p_, const_cast<void *>(sendbuf), GoInt(count),
                             dtype, recvbuf, const_cast<char *>(name), nullptr);
}

int Peer::AllGather(const void *sendbuf, int count, KungFu_Datatype dtype,
                    void *recvbuf, const char *name, const DoneCallback &done)
{
    return GoKungfuAllGather(p_, const_cast<void *>(sendbuf), GoInt(count),
                             dtype, recvbuf, const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int Peer::Broadcast(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, const char *name)
{
    return GoKungfuBroadcast(p_, const_cast<void *>(sendbuf), recvbuf,
                             GoInt(count), dtype, const_cast<char *>(name),
                             nullptr);
}

int Peer::Broadcast(const void *sendbuf, void *recvbuf, int count,
                    KungFu_Datatype dtype, const char *name,
                    const DoneCallback &done)
{
    return GoKungfuBroadcast(p_, const_cast<void *>(sendbuf), recvbuf,
                             GoInt(count), dtype, const_cast<char *>(name),
                             new CallbackWrapper(done));
}

int Peer::LocalBroadcast(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, const char *name)
{
    return GoKungfuLocalBroadcast(p_, const_cast<void *>(sendbuf), recvbuf,
                                  GoInt(count), dtype, const_cast<char *>(name),
                                  nullptr);
}

int Peer::LocalBroadcast(const void *sendbuf, void *recvbuf, int count,
                         KungFu_Datatype dtype, const char *name,
                         const DoneCallback &done)
{
    return GoKungfuLocalBroadcast(p_, const_cast<void *>(sendbuf), recvbuf,
                                  GoInt(count), dtype, const_cast<char *>(name),
                                  new CallbackWrapper(done));
}

int Peer::Gather(const void *sendbuf, int send_count,
                 KungFu_Datatype send_dtype, void *recvbuf, int recv_count,
                 KungFu_Datatype recv_dtype, const char *name)
{
    return GoKungfuGather(p_, const_cast<void *>(sendbuf), GoInt(send_count),
                          send_dtype, recvbuf, GoInt(recv_count), recv_dtype,
                          const_cast<char *>(name), nullptr);
}

int Peer::Gather(const void *sendbuf, int send_count,
                 KungFu_Datatype send_dtype, void *recvbuf, int recv_count,
                 KungFu_Datatype recv_dtype, const char *name,
                 const DoneCallback &done)
{
    return GoKungfuGather(p_, const_cast<void *>(sendbuf), GoInt(send_count),
                          send_dtype, recvbuf, GoInt(recv_count), recv_dtype,
                          const_cast<char *>(name), new CallbackWrapper(done));
}

int Peer::AllGatherTransform(const void *input, int input_count,
                             KungFu_Datatype input_dtype,  //
                             void *output, int output_count,
                             KungFu_Datatype output_dtype,  //
                             const char *name, const TransformFunc &f)
{
    const bool is_root = Rank() == 0;  // FIXME: make sure 0 is root
    if (is_root) {
        const int total_count = Size() * input_count;
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

int Peer::SetTree(const int32_t *tree)
{
    return GoKungfuSetTree(p_, const_cast<int32_t *>(tree));
}
}  // namespace kungfu
