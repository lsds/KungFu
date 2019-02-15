#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>
#include <kungfu_base.h>
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

class AllReduce : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world.AllReduce(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            name().c_str(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduce").Device(DEVICE_CPU), AllReduce);

// Ako implementation
class AkoNegotiator : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // Check arg count: gradient tensors group, number partitions, current
        // partition index
        DCHECK_EQ(5, context->num_inputs());

        const Tensor &partitionTensor       = context->input(0);
        const Tensor &allGradients          = context->input(1);    

        const Tensor &currentPartitionIndex = context->input(2);
        const Tensor &pAkoPartitions        = context->input(3);

        const Tensor &kickinTime            = context->input(4);
        
        Tensor *output                      = nullptr;
        

        auto currentPartitionIndexTensor = currentPartitionIndex.vec<int>();
        auto numberPartitionsTensor      = pAkoPartitions.vec<int>();
        auto kickinTimeTensor            = kickinTime.vec<int>();

        int numberPartitions = numberPartitionsTensor(0);
        int partitionIndex   = currentPartitionIndexTensor(0);
        int kickin           = kickinTimeTensor(0);


        if(_kungfu_world.GetGlobalStep() < kickin) {
            OP_REQUIRES_OK(context,
                           context->allocate_output(0, allGradients.shape(), &output));
        } else {
            OP_REQUIRES_OK(context,
                           context->allocate_output(0, partitionTensor.shape(), &output));
        }

        //std::cout << "Global step: " << _kungfu_world.GetGlobalStep() << std::endl;
       // std::cout << "Kick   step: " << kickin << std::endl;

        if(_kungfu_world.GetGlobalStep() < kickin) {
           // std::cout << "PLAIN NEGOTIATION" << std::endl;
            // perform plain all-reduce until weight updates stabilize to minimize loss
            _kungfu_world.AllReduce(allGradients.tensor_data().data(),
                                    (void *)(output->tensor_data().data()),
                                    allGradients.NumElements(),
                                    to_kungfu_type(allGradients.dtype()), KungFu_SUM,
                                    name().c_str(), done);
        } else if (_kungfu_world.GetGlobalStep() % numberPartitions ==
            partitionIndex) {
           // std::cout << "AKO NEGOTIATION" << std::endl;
            _kungfu_world.AllReduce(partitionTensor.tensor_data().data(),
                                    (void *)(output->tensor_data().data()),
                                    partitionTensor.NumElements(),
                                    to_kungfu_type(partitionTensor.dtype()), KungFu_SUM,
                                    name().c_str(), done);
        } else {
            done();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_CPU),
                        AkoNegotiator);

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
