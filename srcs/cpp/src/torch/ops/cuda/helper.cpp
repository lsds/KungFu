#include <kungfu/cuda/stream.hpp>
#include <kungfu/torch/common.hpp>

namespace kungfu
{
TorchCudaHelper::TorchCudaHelper()
    : up_stream_(new CudaStream), down_stream_(new CudaStream)
{
}

HandleManager<int> &TorchCudaHelper::handle_manager() { return hm_; }

void TorchCudaHelper::from_cuda(void *buffer, const torch::Tensor &t)
{
    down_stream_->memcpy(buffer, t.data_ptr(), data_size(t),
                         cudaMemcpyDeviceToHost);
}

void TorchCudaHelper::from_cuda(void *dst, const void *src, size_t size)
{
    down_stream_->memcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void TorchCudaHelper::to_cuda(torch::Tensor &t, const void *buffer)
{
    up_stream_->memcpy(t.data_ptr(), buffer, data_size(t),
                       cudaMemcpyHostToDevice);
}

void TorchCudaHelper::to_cuda(void *dst, const void *src, size_t size)
{
    up_stream_->memcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void wait_handle(int handle)
{
    _torch_cuda_helper.handle_manager().wait(handle);
}

void wait_all_handles(const std::vector<int> &handles)
{
    _torch_cuda_helper.handle_manager().wait_all(handles);
}

TorchCudaHelper _torch_cuda_helper;
}  // namespace kungfu
