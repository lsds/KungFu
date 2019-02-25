#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_ops.h>
#include <queue>


static kungfu_world _kungfu_world;

namespace tensorflow
{

KungFu_Datatype to_kungfu_type(const DataType &dtype)
{
    switch (dtype) {
    case DT_INT32:
        return KungFu_INT32;
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
    // have them public for now

    std::queue<Tensor> tensorWindow;
    Tensor runningSum;

    explicit AkoNegotiator(OpKernelConstruction* context) : AsyncOpKernel(context) {
        std::cout << "Initializing ako negotiator " << context->input_type(0) << std::endl;

    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // Check arg count: gradient tensors group, number partitions, current
        // partition index
        DCHECK_EQ(4, context->num_inputs());

        
        const Tensor &gradients          = context->input(0);    

        const Tensor &currentPartitionIndex = context->input(1);
        const Tensor &pAkoPartitions        = context->input(2);

        const Tensor &kickinTime            = context->input(3);
        
        Tensor *output                      = nullptr;

        auto currentPartitionIndexTensor = currentPartitionIndex.vec<int>();
        auto numberPartitionsTensor      = pAkoPartitions.vec<int>();
        auto kickinTimeTensor            = kickinTime.vec<int>();

        int numberPartitions = numberPartitionsTensor(0);
        int partitionIndex   = currentPartitionIndexTensor(0);
        int kickin           = kickinTimeTensor(0);

        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        // FIXME
        auto flt = output->flat<float>();
        for (int i = 0; i < flt.size(); ++i) {
            flt(i) = 0.0;
        }
        
        if (_kungfu_world.GetGlobalStep() % numberPartitions == partitionIndex) {           
            if(runningSum.NumElements() == 0) {
                Tensor grad_accumulated(DataTypeToEnum<float>::v(), gradients.shape());
                auto grad_accumulated_flt = grad_accumulated.flat<float>();
                for (int i = 0; i < grad_accumulated_flt.size(); ++i) {
                    grad_accumulated_flt(i) = 0.0;
                }
               runningSum = grad_accumulated;
            }

            auto grads_flt = gradients.flat<float>();
            auto runningSum_flt = runningSum.flat<float>();
            for (int i = 0; i < runningSum_flt.size(); ++i) {
                runningSum_flt(i) = grads_flt(i) + runningSum_flt(i);
            }

            // TODO: combine with prev loop
            tensorWindow.push(gradients);    
            if(tensorWindow.size() > numberPartitions) {
               Tensor stale = tensorWindow.front();
               auto stale_flt = stale.flat<float>();
               for(int i = 0; i < runningSum_flt.size(); ++i) {
                   runningSum_flt(i) = runningSum_flt(i) - stale_flt(i);
               }
               tensorWindow.pop();
            }

            // Compute running sum average (CHECK)
            // Tensor avg(DataTypeToEnum<float>::v(), runningSum.shape());
            // auto avg_flt = avg.flat<float>();

            // for (int i = 0; i < avg_flt.size(); ++i) {
            //     avg_flt(i) = 0.0;
            // }

            // for (int i = 0; i < runningSum_flt.size(); ++i) {
            //     avg_flt(i) = runningSum_flt(i) / tensorWindow.size();
            // }


            _kungfu_world.AllReduce(runningSum.tensor_data().data(),
                                    (void *)(output->tensor_data().data()),
                                    runningSum.NumElements(),
                                    to_kungfu_type(runningSum.dtype()), KungFu_SUM,
                                    name().c_str(), done); // give it an empty callback to make it async
        } else {
            done();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_CPU),
                        AkoNegotiator);

class Broadcast : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;
    
  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {                        
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world.Broadcast(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), name().c_str(),
            done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), Broadcast);

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

class GlobalVariance : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        // TODO
    }
};

REGISTER_KERNEL_BUILDER(Name("GlobalVariance").Device(DEVICE_CPU),
                        GlobalVariance);

}  // namespace tensorflow
