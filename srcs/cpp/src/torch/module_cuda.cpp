#include <torch/extension.h>

namespace kungfu
{
void all_reduce_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type, const std::string &op);

void all_reduce_cuda(torch::Tensor input, torch::Tensor output,
                     const std::string &type, const std::string &op_name);
}  // namespace kungfu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce_cpu", &kungfu::all_reduce_cpu);    //
    m.def("all_reduce_cuda", &kungfu::all_reduce_cuda);  //
}
