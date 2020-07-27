#include <iostream>
#include <vector>

#include <torch/extension.h>

#include <kungfu/torch/common.hpp>

namespace kungfu
{
void all_reduce_cuda(torch::Tensor input, torch::Tensor output,
                     const std::string &type, const std::string &op_name)
{
    const auto tt     = _torch_tensor_types.at(type);
    TensorShape shape = get_tensor_shape(input);
    DBG(std::string(__func__) + " " + type + "" + " called with dtype " +
        std::to_string(int(input.scalar_type())) + " shape: " + shape.str());
    if (tt == Torch_Cuda_Float) {
        std::cerr << "TODO: " << __func__ << std::endl;
    } else {
    }
}
}  // namespace kungfu
