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

class Negotiator : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world.Negotiate(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            name().c_str(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_CPU), Negotiator);


// Ako implementation
class AkoNegotiator : public AsyncOpKernel
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

        auto currentPartitionIndexTensor = currentPartitionIndex.vec<int>();
        auto numberPartitionsTensor = pAkoPartitions.vec<int>();

        int numberPartitions = numberPartitionsTensor(0);
        int partitionIndex   = currentPartitionIndexTensor(0);

        // This should be the total number of nodes, not the global step
        // if p > NumberOfNodes: a node receives multiple partitions
        // if p == NumberOfNodes: all nodes receive exactly one partition
        // if p < NumberOfNodes: some nodes do not receive the partition at all
        if(_kungfu_world.GetGlobalStep() % numberPartitions == partitionIndex) {
          _kungfu_world.Negotiate(
              input.tensor_data().data(), (void *)(output->tensor_data().data()),
              input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
              name().c_str(), done);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_CPU),
                        AkoNegotiator);

#if KUNGFU_HAVE_GPU
REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_GPU),
                        AkoNegotiator);
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

class SetNumGradients : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        int32_t *x = static_cast<int32_t *>((void *)input.tensor_data().data());
        _kungfu_world.SetNumGradients(x[0]);
    }
};

REGISTER_KERNEL_BUILDER(Name("SetNumGradients").Device(DEVICE_CPU),
                        SetNumGradients);

}  // namespace tensorflow
