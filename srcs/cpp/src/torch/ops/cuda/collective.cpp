#include <iostream>
#include <vector>

#include <torch/extension.h>

#include <kungfu/torch/common.hpp>

// TODO
void cuda_all_reduce(torch::Tensor input, torch::Tensor output,
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
