#include <kungfu/cuda/stream.hpp>
#include <kungfu/torch/common.hpp>

namespace kungfu
{
TorchCudaHelper::TorchCudaHelper() : stream_(new CudaStream) {}

void TorchCudaHelper::from_cuda(void *buffer, const torch::Tensor &input)
{
    //
}

void TorchCudaHelper::to_cuda(torch::Tensor &input, const void *buffer)
{
    //
}

TorchCudaHelper _torch_cuda_helper;
}  // namespace kungfu
