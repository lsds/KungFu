#include <torch/extension.h>

namespace kungfu
{
void all_reduce_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type, const std::string &op);

void all_gather_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type);
}  // namespace kungfu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce_cpu", &kungfu::all_reduce_cpu);
    m.def("all_gather_cpu", &kungfu::all_gather_cpu);
}
