#include <torch/extension.h>

void all_reduce(torch::Tensor input, torch::Tensor output,
                const std::string &type, const std::string &op);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce", &all_reduce, "all_reduce");
}
