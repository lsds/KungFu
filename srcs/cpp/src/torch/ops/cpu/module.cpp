#include <torch/extension.h>

std::vector<at::Tensor> all_reduce(torch::Tensor input);
// int all_reduce_async(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce", &all_reduce, "all_reduce");
    // m.def("all_reduce_async", &all_reduce_async, "all_reduce_async");
}
