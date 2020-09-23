#include <iostream>
#include <vector>

#include <torch/extension.h>

#include <kungfu/torch/common.hpp>

namespace kungfu
{
template <typename T>
void do_all_reduce(void *input, void *output, size_t n, KungFu_Op op)
{
    T *x = reinterpret_cast<T *>(input);
    T *y = reinterpret_cast<T *>(output);
    _default_peer->AllReduce(x, y, n, kungfu::type_encoder::value<T>(), op, "");
}

void do_all_reduce(torch::Tensor &input, torch::Tensor &output, KungFu_Op op)
{
    using T           = float;
    TensorShape shape = get_tensor_shape(input);
    do_all_reduce<T>(input.data_ptr(), output.data_ptr(), shape.size(), op);
}

void all_reduce_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type, const std::string &op_name)
{
    const auto tt     = _torch_tensor_types.at(type);
    TensorShape shape = get_tensor_shape(input);
    if (tt == Torch_Cpu_Float) {
        do_all_reduce(input, output, _kungfu_ops.at(op_name));
    } else {
        std::cerr << __func__ << " not implemented for " << type << std::endl;
    }
}

template <typename T>
void do_all_gather(void *input, void *output, size_t n)
{
    T *x = reinterpret_cast<T *>(input);
    T *y = reinterpret_cast<T *>(output);
    _default_peer->AllGather(x, n, kungfu::type_encoder::value<T>(), y, "");
}

void do_all_gather(torch::Tensor &input, torch::Tensor &output)
{
    using T           = float;
    TensorShape shape = get_tensor_shape(input);
    do_all_gather<T>(input.data_ptr(), output.data_ptr(), shape.size());
}

void all_gather_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type)
{
    const auto tt     = _torch_tensor_types.at(type);
    TensorShape shape = get_tensor_shape(input);
    if (tt == Torch_Cpu_Float) {
        do_all_gather(input, output);
    } else {
        std::cerr << __func__ << " not implemented for " << type << std::endl;
    }
}
}  // namespace kungfu
