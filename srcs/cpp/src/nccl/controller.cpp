#include <kungfu/cuda/stream.hpp>
#include <kungfu/nccl/controller.hpp>
#include <kungfu/nccl/helper.hpp>

namespace kungfu
{
void CrossAllReduceGpu(Peer *peer, const Workspace &w, KungFu_Op op,
                       const std::string &name, DoneCallback done)
{
    if (peer->LocalRank() != 0) {
        peer->Noop(done);
        return;
    }
    const int data_size = w.count * kungfu_type_size(w.dtype);
    if (peer->HostCount() <= 1) {
        if (w.sendbuf != w.recvbuf) {
            CudaStream stream;
            stream.memcpy(w.recvbuf, w.sendbuf, data_size,
                          cudaMemcpyDeviceToDevice);
        }
        peer->Noop(done);
        return;
    }
    char *buffer = new char[data_size];
    {
        CudaStream stream;
        stream.memcpy(buffer, w.sendbuf, data_size, cudaMemcpyDeviceToHost);
    }
    peer->CrossAllReduce(buffer, buffer, w.count, w.dtype, op, name.c_str(),
                         [=] {
                             {
                                 CudaStream stream;
                                 stream.memcpy(w.recvbuf, buffer, data_size,
                                               cudaMemcpyHostToDevice);
                             }
                             delete[] buffer;
                             peer->Noop(done);
                         });
}

NCCLController::NCCLController(const KungFu_NCCLScope scope) : scope_(scope) {}

void NCCLController::InitOnce(Peer *peer)
{
    if (gpu_collective_.get() == nullptr) {
        if (scope_ == KungFu_NCCL_LOCAL) {
            gpu_collective_.reset(new_local_gpu_collective(*peer));
        } else {
            gpu_collective_.reset(new_global_gpu_collective(*peer));
        }
    }
}

int NCCLController::Reduce(const Workspace &w, KungFu_Op op, DoneCallback done)
{
    gpu_collective_->reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

int NCCLController::Broadcast(const Workspace &w, DoneCallback done)
{
    gpu_collective_->broadcast(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

int NCCLController::AllReduce(const Workspace &w, KungFu_Op op,
                              DoneCallback done)
{
    gpu_collective_->all_reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}
}  // namespace kungfu
