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
    using T = float;
    std::cerr << __func__ << std::endl;
    TensorShape shape = get_tensor_shape(input);
    do_all_reduce<T>(input.data_ptr(), output.data_ptr(), shape.size(), op);
}

void all_reduce_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type, const std::string &op_name)
{
    const auto tt     = _torch_tensor_types.at(type);
    TensorShape shape = get_tensor_shape(input);
    DBG(std::string(__func__) + " " + type + "" + " called with dtype " +
        std::to_string(int(input.scalar_type())) + " shape: " + shape.str());
    if (tt == Torch_Cpu_Float) {
        do_all_reduce(input, output, _kungfu_ops.at(op_name));
    } else {
        std::cerr << __func__ << " not implemented for " << type << std::endl;
    }
}
}  // namespace kungfu
