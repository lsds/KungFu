#include <iostream>
#include <vector>

#include <torch/extension.h>

#include <kungfu/torch/common.hpp>

namespace kungfu
{
KungFu_Datatype from(Torch_Tensor_Type tt)
{
    switch (tt) {
    case Torch_Cuda_Float:
        return KungFu_FLOAT;
    default:
        throw std::runtime_error("Not implemented");
    }
}

void all_reduce_cuda(torch::Tensor input, torch::Tensor output,
                     const std::string &type, const std::string &op_name)
{
    // FIXME: make sure input and output are ready
    const auto tt               = _torch_tensor_types.at(type);
    const KungFu_Datatype dtype = from(tt);
    TensorShape shape           = get_tensor_shape(input);
    const size_t count          = shape.size();
    if (tt == Torch_Cuda_Float) {
        std::vector<char> buffer(count * kungfu_type_size(dtype));
        _torch_cuda_helper.from_cuda(buffer.data(), input);
        _default_peer->AllReduce(buffer.data(), buffer.data(), count, dtype,
                                 _kungfu_ops.at(op_name), "");
        _torch_cuda_helper.to_cuda(output, buffer.data());
    } else {
        throw std::runtime_error(__func__ +
                                 std::string(" NOT implemented for ") + type);
    }
}
}  // namespace kungfu
