#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_ops.h>
#include <queue>

#include <chrono>


static kungfu_world _kungfu_world;

namespace tensorflow
{

KungFu_Datatype to_kungfu_type(const DataType &dtype)
{
    switch (dtype) {
    case DT_INT32:
        return KungFu_INT32;
    case DT_INT64:
        return KungFu_INT64;
    case DT_FLOAT:
        return KungFu_FLOAT;
    case DT_DOUBLE:
        return KungFu_DOUBLE;
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

typedef std::function<void(Tensor&, Tensor&)> AccumulateExternalGradientsCallback;

void accumulateExternalFunc(Tensor& accumulatedLocalGradients, Tensor& gradients) {

}


// Ako implementation
class AkoNegotiator : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    // have them public for now

    std::queue<Tensor> localGradientsWindow;
    Tensor accumulatedLocalGradients;
    Tensor accumulatedExternalGradients;

     explicit OpKernel(OpKernelConstruction* context) {
        // accummulateExternalCb =    
     }


    void Compute(OpKernelContext *context) override
    {
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


        localGradientsWindow.push(gradients);    

        Tensor stale;
        if(localGradientsWindow.size() > numberPartitions) {
            stale = localGradientsWindow.front();
            localGradientsWindow.pop();
        }
    
        if (_kungfu_world.GetGlobalStep() % numberPartitions == partitionIndex) {           
            if(accumulatedLocalGradients.NumElements() == 0) {
                Tensor grad_accumulated(DataTypeToEnum<float>::v(), gradients.shape());
                auto grad_accumulated_flt = grad_accumulated.flat<float>();
                for (int i = 0; i < grad_accumulated_flt.size(); ++i) {
                    grad_accumulated_flt(i) = 0.0;
                }
               accumulatedLocalGradients = grad_accumulated;
            }

            auto grads_flt = gradients.flat<float>();
            auto accumulatedLocalGradients_flt = accumulatedLocalGradients.flat<float>();
            
            auto expire = stale.NumElements() > 0;
            for (int i = 0; i < accumulatedLocalGradients_flt.size(); ++i) {
                auto expiredTensor = expire ? stale.flat<float>()(i) : 0.0;
                accumulatedLocalGradients_flt(i) = grads_flt(i) + accumulatedLocalGradients_flt(i) - expiredTensor;
            }

            _kungfu_world.AllReduce(accumulatedLocalGradients.tensor_data().data(),
                                    (void *)(output->tensor_data().data()),
                                    accumulatedLocalGradients.NumElements(),
                                    to_kungfu_type(accumulatedLocalGradients.dtype()), KungFu_SUM,
                                    name().c_str(), accumulateExternalCb); // give it an empty callback to make it async
        } else {
            auto flt = output->flat<float>();
            for (int i = 0; i < flt.size(); ++i) {
                flt(i) = 0.0;
            }
        }
    }

    private:
       AccumulateExternalGradientsCallback accummulateExternalCb;
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
