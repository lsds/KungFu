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
    const KungFu_Datatype dtype = from(_torch_tensor_types.at(type));
    std::vector<char> buffer(data_size(input));
    _torch_cuda_helper.from_cuda(buffer.data(), input);
    _default_peer->AllReduce(buffer.data(), buffer.data(), input.numel(), dtype,
                             _kungfu_ops.at(op_name), "");
    _torch_cuda_helper.to_cuda(output, buffer.data());
}

int all_reduce_cuda_async(torch::Tensor input, torch::Tensor output,
                          const std::string &type, const std::string &op_name,
                          const std::string &tensor_name)
{
    const KungFu_Datatype dtype = from(_torch_tensor_types.at(type));
    const auto count            = input.numel();
    const auto size             = data_size(input);
    const auto op               = _kungfu_ops.at(op_name);
    const void *px              = input.data_ptr();
    void *py                    = output.data_ptr();

    const int handle = _torch_cuda_helper.handle_manager().create();
    _default_peer->Noop([=] {
        char *buffer = new char[size];
        _torch_cuda_helper.from_cuda(buffer, px, size);
        _default_peer->AllReduce(
            buffer, buffer, count, dtype, op, tensor_name.c_str(), [=] {
                _torch_cuda_helper.to_cuda(py, buffer, size);
                delete[] buffer;
                _torch_cuda_helper.handle_manager().done(handle);
            });
    });
    return handle;
}

int broadcast_cuda_async(torch::Tensor input, torch::Tensor output,
                         const std::string &type,
                         const std::string &tensor_name)
{
    const KungFu_Datatype dtype = from(_torch_tensor_types.at(type));
    const auto count            = input.numel();
    const auto size             = data_size(input);
    const void *px              = input.data_ptr();
    void *py                    = output.data_ptr();

    const int handle = _torch_cuda_helper.handle_manager().create();
    _default_peer->Noop([=] {
        char *buffer = new char[size];
        _torch_cuda_helper.from_cuda(buffer, px, size);
        _default_peer->Broadcast(
            buffer, buffer, count, dtype, tensor_name.c_str(), [=] {
                _torch_cuda_helper.to_cuda(py, buffer, size);
                delete[] buffer;
                _torch_cuda_helper.handle_manager().done(handle);
            });
    });
    return handle;
}
}  // namespace kungfu
