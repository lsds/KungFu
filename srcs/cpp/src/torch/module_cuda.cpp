#include <torch/extension.h>

namespace kungfu
{
void all_reduce_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type, const std::string &op_name);

void all_reduce_cuda(torch::Tensor input, torch::Tensor output,
                     const std::string &type, const std::string &op_name);

int all_reduce_cuda_async(torch::Tensor input, torch::Tensor output,
                          const std::string &type, const std::string &op_name,
                          const std::string &tensor_name);

int broadcast_cuda_async(torch::Tensor input, torch::Tensor output,
                         const std::string &type,
                         const std::string &tensor_name);

void wait_handle(int handle);
void wait_all_handles(const std::vector<int> &handles);

void all_gather_cpu(torch::Tensor input, torch::Tensor output,
                    const std::string &type);
void all_gather_cuda(torch::Tensor input, torch::Tensor output,
                     const std::string &type);
}  // namespace kungfu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce_cpu", &kungfu::all_reduce_cpu);
    m.def("all_reduce_cuda", &kungfu::all_reduce_cuda);
    m.def("all_reduce_cuda_async", &kungfu::all_reduce_cuda_async);
    m.def("broadcast_cuda_async", &kungfu::broadcast_cuda_async);

    m.def("wait_handle", &kungfu::wait_handle);
    m.def("wait_all_handles", &kungfu::wait_all_handles);

    m.def("all_gather_cpu", &kungfu::all_gather_cpu);
    m.def("all_gather_cuda", &kungfu::all_gather_cuda);
}
