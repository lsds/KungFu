#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/python/c_api.h>
#include <kungfu/utils/handler_manager.hpp>
#include <torch/extension.h>

namespace kungfu
{
inline size_t data_size(const torch::Tensor &t)
{
    return t.element_size() * t.numel();
}

class TensorShape
{
    std::vector<int64_t> dims_;

  public:
    void AddDim(int d);

    std::string str() const;

    const std::vector<int64_t> &Dims() const;

    int64_t size() const;
};

TensorShape get_tensor_shape(torch::Tensor &x);

torch::Tensor new_tensor_like(torch::Tensor input);

void do_all_reduce(torch::Tensor &input, torch::Tensor &output,
                   KungFu_Op op = KungFu_SUM);

class CudaStream;

class TorchCudaHelper
{
    std::unique_ptr<CudaStream> up_stream_;
    std::unique_ptr<CudaStream> down_stream_;
    HandleManager<int> hm_;

  public:
    TorchCudaHelper();

    HandleManager<int> &handle_manager();

    void from_cuda(void *buffer, const torch::Tensor &t);
    void from_cuda(void *dst, const void *src, size_t size);
    void to_cuda(torch::Tensor &t, const void *buffer);
    void to_cuda(void *dst, const void *src, size_t size);
};

void wait_handle(int handle);

extern TorchCudaHelper _torch_cuda_helper;

enum Torch_Tensor_Type {
    Torch_Cpu_Float,
    Torch_Cuda_Float,
};

extern const std::map<std::string, KungFu_Op> _kungfu_ops;

extern const std::map<std::string, Torch_Tensor_Type> _torch_tensor_types;

void DBG(const std::string &msg);
}  // namespace kungfu
