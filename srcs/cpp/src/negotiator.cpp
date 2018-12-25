#include <negotiator.h>

#include <thread>

#include <tensorflow/core/framework/op_kernel.h>
#if KUNGFU_HAVE_GPU
#include <cuda_runtime.h>
#endif
#include <kungfu.hpp>

class _kungfu_t
{

  public:
    _kungfu_t() { KungfuInit(); }

    ~_kungfu_t() { KungfuFinalize(); }
};

static _kungfu_t _kungfu_world;

namespace tensorflow
{

int to_kungfu_type(const DataType &dtype)
{
    switch (dtype) {
    case DT_FLOAT:
        return KungFu_FLOAT;
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

// TODO: use existing API
int type_size(const DataType &dtype)
{
    switch (dtype) {
    case DT_FLOAT:
        return 4;
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device> struct NegotiatorImpl;

template <> struct NegotiatorImpl<CPUDevice> {
    void operator()(const void *input, void *output, int n,
                    const tensorflow::DataType &dtype, const std::string &name,
                    DoneCallback done) const
    {
        KungfuNegotiateAsync(input, output, n, to_kungfu_type(dtype),
                             KungFu_SUM, name.c_str(), done);
    }
};

#if KUNGFU_HAVE_GPU
template <> struct NegotiatorImpl<GPUDevice> {
    void operator()(const void *input, void *output, int n,
                    const tensorflow::DataType &dtype, const std::string &name,
                    DoneCallback done) const
    {
        const int buffer_size = type_size(dtype) * n;
        // TODO: use memory pool
        auto input_cpu = new std::vector<char>(buffer_size);
        auto output_cpu = new std::vector<char>(buffer_size);

        if (cudaMemcpy(input_cpu->data(), input, buffer_size,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            LOG(FATAL) << "cudaMemcpy failed";
        }
        KungfuNegotiateAsync(
            input_cpu->data(), output_cpu->data(), n, to_kungfu_type(dtype),
            KungFu_SUM, name.c_str(),
            [done, input_cpu, output_cpu, output, buffer_size] {
                if (cudaMemcpy(output, output_cpu->data(), buffer_size,
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    LOG(FATAL) << "cudaMemcpy failed";
                }
                delete input_cpu;
                delete output_cpu;
                done();
            });
    }
};
#endif

template <typename Device> class Negotiator : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        NegotiatorImpl<Device>()(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), input.dtype(), name(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_CPU),
                        Negotiator<CPUDevice>);

#if KUNGFU_HAVE_GPU
REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_GPU),
                        Negotiator<GPUDevice>);
#endif

}  // namespace tensorflow
