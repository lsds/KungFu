#include <kungfu.h>
#include <kungfu/python/init.h>
#include <kungfu/utils/cuda_helper.hpp>

std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;

void kungfu_python_init_nccl()
{
    _default_nccl_helper.reset(new kungfu::NCCLHelper);
}

void kungfu_python_finialize_nccl() { _default_nccl_helper.reset(nullptr); }

namespace kungfu
{
void CrossAllReduceGpu(const Workspace &w, KungFu_Op op,
                       const std::string &name, DoneCallback done)
{
    if (_default_peer->LocalRank() != 0) {
        _default_peer->Noop(done);
        return;
    }
    const int data_size = w.count * kungfu_type_size(w.dtype);
    if (_default_peer->HostCount() <= 1) {
        if (w.sendbuf != w.recvbuf) {
            CudaStream stream;
            stream.memcpy(w.recvbuf, w.sendbuf, data_size,
                          cudaMemcpyDeviceToDevice);
        }
        _default_peer->Noop(done);
        return;
    }
    char *buffer = new char[data_size];
    CudaStream stream;
    stream.memcpy(buffer, w.sendbuf, data_size, cudaMemcpyDeviceToHost);
    _default_peer->CrossAllReduce(
        buffer, buffer, w.count, w.dtype, op, name.c_str(), [=] {
            CudaStream stream;
            stream.memcpy(w.recvbuf, buffer, data_size, cudaMemcpyHostToDevice);
            delete[] buffer;
            _default_peer->Noop(done);
        });
}

NCCLController::NCCLController(const KungFu_NCCLScope scope) : scope_(scope) {}

void NCCLController::InitOnce()
{
    if (gpu_collective_.get() == nullptr) {
        if (scope_ == KungFu_NCCL_LOCAL) {
            gpu_collective_.reset(new_local_gpu_collective(*_default_peer));
        } else {
            gpu_collective_.reset(new_global_gpu_collective(*_default_peer));
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

NCCLController *NCCLHelper::EnsureController(const KungFu_NCCLScope scope)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = controllers_[scope];
    if (ptr.get() == nullptr) {
        ptr.reset(new NCCLController(scope));
        ptr->InitOnce();
    }
    return ptr.get();
}

NCCLScheduler *NCCLHelper::EnsureScheduler(const KungFu_NCCLScope scope)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = schedulers_[scope];
    if (ptr.get() == nullptr) { ptr.reset(new NCCLScheduler(scope)); }
    return ptr.get();
}
}  // namespace kungfu
