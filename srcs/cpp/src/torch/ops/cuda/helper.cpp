#include <kungfu/cuda/stream.hpp>
#include <kungfu/torch/common.hpp>

namespace kungfu
{
TorchCudaHelper::TorchCudaHelper() : stream_(new CudaStream) {}

void TorchCudaHelper::from_cuda(void *buffer, const torch::Tensor &t)
{
    stream_->memcpy(buffer, t.data_ptr(), t.element_size() * t.numel(),
                    cudaMemcpyDeviceToHost);
}

void TorchCudaHelper::to_cuda(torch::Tensor &t, const void *buffer)
{
    stream_->memcpy(t.data_ptr(), buffer, t.element_size() * t.numel(),
                    cudaMemcpyHostToDevice);
}

TorchCudaHelper _torch_cuda_helper;
}  // namespace kungfu
