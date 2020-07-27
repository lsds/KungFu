#pragma once
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/python/init.h>
#include <torch/extension.h>

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

enum Torch_Tensor_Type {
    Torch_Cpu_Float,
    Torch_Cuda_Float,
};

extern const std::map<std::string, KungFu_Op> _kungfu_ops;

extern const std::map<std::string, Torch_Tensor_Type> _torch_tensor_types;

void DBG(const std::string &msg);
