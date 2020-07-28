#include <kungfu/cuda/stream.hpp>
#include <kungfu/torch/common.hpp>

namespace kungfu
{
TorchCudaHelper::TorchCudaHelper() : stream_(new CudaStream) {}

HandleManager<int> &TorchCudaHelper::handle_manager() { return hm_; }

void TorchCudaHelper::from_cuda(void *buffer, const torch::Tensor &t)
{
    stream_->memcpy(buffer, t.data_ptr(), data_size(t), cudaMemcpyDeviceToHost);
}

void TorchCudaHelper::from_cuda(void *dst, const void *src, size_t size)
{
    stream_->memcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void TorchCudaHelper::to_cuda(torch::Tensor &t, const void *buffer)
{
    stream_->memcpy(t.data_ptr(), buffer, data_size(t), cudaMemcpyHostToDevice);
}

void TorchCudaHelper::to_cuda(void *dst, const void *src, size_t size)
{
    stream_->memcpy(dst, src, size, cudaMemcpyHostToDevice);
}

TorchCudaHelper _torch_cuda_helper;
}  // namespace kungfu
