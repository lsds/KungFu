#include <thread>

#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>
#include <kungfu_tensorflow_ops.h>

static kungfu_world _kungfu_world;

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
        _kungfu_world.NegotiateAsync(input, output, n, dtype, KungFu_SUM,
                                     name.c_str(), done);
    }
};

#if KUNGFU_HAVE_GPU
template <> struct NegotiatorImpl<GPUDevice> {
    void operator()(const void *input, void *output, int n,
                    const KungFu_Datatype dtype, const std::string &name,
                    DoneCallback done) const
    {
        _kungfu_world.NegotiateGPUAsync(input, output, n, dtype, KungFu_SUM,
                                        name.c_str(), done);
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


// Ako implementation
template <typename Device> class AkoNegotiator : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // Check arg count: gradient tensors group, number partitions, current
        // partition index
        DCHECK_EQ(3, context->num_inputs());

        const Tensor &input                  = context->input(0);
        const Tensor &currentPartitionIndex  = context->input(1);
        const Tensor &pAkoPartitions         = context->input(2);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        // only do the negotiation if the global step mod p is 1
        // this is within a similar tf operator

        // one where I group by tensor size => level out group sizes
        // one other strategy
        int numberPartitions = pAkoPartitions.tensor_data().data();
        int partitionIndex   = currentPartitionIndex.tensor_data().data();
        if(_kungfu_world._global_step % numberPartitions == partitionIndex) {
          NegotiatorImpl<Device>()(
              input.tensor_data().data(), (void *)(output->tensor_data().data()),
              input.NumElements(), to_kungfu_type(input.dtype()), name(), done);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_CPU),
                        Negotiator<CPUDevice>);

#if KUNGFU_HAVE_GPU
REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_GPU),
                        Negotiator<GPUDevice>);
#endif

class GlobalStepModifier : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);  // ignore input
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        int32_t *y =
            static_cast<int32_t *>((void *)output->tensor_data().data());
        y[0] = _kungfu_world.AdvanceGlobalStep();
    }
};

REGISTER_KERNEL_BUILDER(Name("GlobalStepModifier").Device(DEVICE_CPU),
                        GlobalStepModifier);

class SetGradientCount : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        int32_t *x = static_cast<int32_t *>((void *)input.tensor_data().data());
        _kungfu_world.SetGradientCount(x[0]);
    }
};

REGISTER_KERNEL_BUILDER(Name("SetGradientCount").Device(DEVICE_CPU),
                        SetGradientCount);

}  // namespace tensorflow
