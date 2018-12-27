#include <negotiator.h>

#include <thread>

#include <tensorflow/core/framework/op_kernel.h>
#if KUNGFU_HAVE_GPU
#include <cuda_runtime.h>
#endif
#include <kungfu.hpp>

class _kungfu_t
{
    static std::string safe_getenv(const char *name)
    {
        const char *ptr = std::getenv(name);
        if (ptr) { return std::string(ptr); }
        return "";
    }

    KungFu_AllReduceAlgo get_algo() const
    {
        const auto value = safe_getenv("KUNGFU_ALLREDUCE_ALGO");
        const std::map<std::string, KungFu_AllReduceAlgo> mp({
            {"SIMPLE", KungFu_SimpleAllReduce},
            {"RING", KungFu_RingAllReduce},
            {"CLIQUE", KungFu_FullSymmetricAllReduce},
            {"TREE", KungFu_TreeAllReduce},
        });
        if (mp.count(value) > 0) { return mp.at(value); }
        return KungFu_SimpleAllReduce;
    }

  public:
    _kungfu_t()
    {
        const auto algo = get_algo();
        LOG(INFO) << "using all reduce algo: " << algo;
        KungfuInit(algo);
    }

    ~_kungfu_t() { KungfuFinalize(); }
};

static _kungfu_t _kungfu_world;

namespace tensorflow
{

KungFu_Datatype to_kungfu_type(const DataType &dtype)
{
    switch (dtype) {
    case DT_FLOAT:
        return KungFu_FLOAT;
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
                    const KungFu_Datatype dtype, const std::string &name,
                    DoneCallback done) const
    {
        KungfuNegotiateAsync(input, output, n, dtype, KungFu_SUM, name.c_str(),
                             done);
    }
};

#if KUNGFU_HAVE_GPU
template <> struct NegotiatorImpl<GPUDevice> {
    void operator()(const void *input, void *output, int n,
                    const KungFu_Datatype dtype, const std::string &name,
                    DoneCallback done) const
    {
        const int buffer_size = kungfu_type_size(dtype) * n;
        // TODO: use memory pool
        auto input_cpu  = new std::vector<char>(buffer_size);
        auto output_cpu = new std::vector<char>(buffer_size);

        if (cudaMemcpy(input_cpu->data(), input, buffer_size,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            LOG(FATAL) << "cudaMemcpy failed";
        }
        KungfuNegotiateAsync(
            input_cpu->data(), output_cpu->data(), n, dtype, KungFu_SUM,
            name.c_str(), [done, input_cpu, output_cpu, output, buffer_size] {
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
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        NegotiatorImpl<Device>()(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), name(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_CPU),
                        Negotiator<CPUDevice>);

#if KUNGFU_HAVE_GPU
REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_GPU),
                        Negotiator<GPUDevice>);
#endif

}  // namespace tensorflow
