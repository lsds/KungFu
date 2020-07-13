#include <iostream>
#include <vector>

#include <torch/extension.h>

#include "common.hpp"

template <typename T>
void do_all_reduce(void *input, void *output, size_t n)
{
    T *x = reinterpret_cast<T *>(input);
    T *y = reinterpret_cast<T *>(output);
    _default_peer->AllReduce(x, y, n, kungfu::type_encoder::value<T>(),
                             KungFu_SUM, "");
}

void do_all_reduce(torch::Tensor &input, torch::Tensor &output)
{
    using T = float;
    std::cerr << __func__ << std::endl;
    TensorShape shape = get_tensor_shape(input);
    do_all_reduce<T>(input.data_ptr(), output.data_ptr(), shape.size());
}

std::vector<at::Tensor> all_reduce(torch::Tensor input)
{
    TensorShape shape = get_tensor_shape(input);
    std::cerr << __func__ << " called with dtype: " << input.scalar_type()
              << ", shape:" << shape.str() << ", data: " << input.data_ptr()
              << std::endl;

    torch::Tensor output = new_tensor_like(input);
    do_all_reduce(input, output);
    return {output};
}

// int all_reduce_async(torch::Tensor input, torch::Tensor output)
// {
//     TensorShape in_shape  = get_tensor_shape(input);
//     TensorShape out_shape = get_tensor_shape(output);
//     std::cerr << __func__ << " called with dtype: " << input.scalar_type()
//               << ", shape:" << in_shape.str() << ", data: " <<
//               input.data_ptr()
//               << std::endl;
//     return 0;
// }
