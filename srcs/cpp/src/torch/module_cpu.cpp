#include <torch/extension.h>

namespace kungfu
{
void all_reduce_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type, const std::string &op);
}  // namespace kungfu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce_cpu", &kungfu::all_reduce_cpu);    //
}
