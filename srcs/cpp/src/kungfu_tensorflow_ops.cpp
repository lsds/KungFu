#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_ops.h>
#include <queue>
#include <mutex>
#include <atomic>

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

// Ako implementation
class AkoNegotiator : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    // have them public for now

    std::queue<Tensor> tensorWindow;
    Tensor outGrad; // the accumulated gradient to be negotiated
    Tensor inGrad;  // the accumulated gradient received through negotiation
    std::mutex inGradMutex;
    std::atomic<bool> hasInGrad;


    explicit AkoNegotiator(OpKernelConstruction* context) : AsyncOpKernel(context) {
        std::cout << "Initializing ako negotiator" << std::endl;
        hasInGrad = true;
        std::cout << "After init ako negotiator" << std::endl;
        
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // Check arg count: gradient tensors group, number partitions, current
        // partition index
        DCHECK_EQ(4, context->num_inputs());

        
        Tensor &gradients             = (Tensor&) context->input(0);    

        const Tensor &currentPartitionIndex = context->input(1);
        const Tensor &pAkoPartitions        = context->input(2);

        const Tensor &kickinTime            = context->input(3);
        
        Tensor *output                      = nullptr;

        auto currentPartitionIndexTensor = currentPartitionIndex.vec<int>();
        auto numberPartitionsTensor      = pAkoPartitions.vec<int>();
        auto kickinTimeTensor            = kickinTime.vec<int>();

        uint32_t numberPartitions = numberPartitionsTensor(0);
        uint32_t partitionIndex   = currentPartitionIndexTensor(0);
        uint32_t kickin           = kickinTimeTensor(0);

        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        // Update gradient window

        tensorWindow.push(gradients);    
        Tensor stale;
        if(tensorWindow.size() > numberPartitions) {
            stale = tensorWindow.front();
            tensorWindow.pop(); 
        }

        if(outGrad.NumElements() == 0) {
            Tensor tensorReset(DataTypeToEnum<float>::v(), gradients.shape());
            Tensor grad_accumulated = tensorReset;
            auto grad_accumulated_flt = grad_accumulated.flat<float>();
            for (int i = 0; i < grad_accumulated_flt.size(); ++i) {
                grad_accumulated_flt(i) = 0.0;
            }
            outGrad = grad_accumulated;
        }



        auto grads_flt   = gradients.flat<float>();
        auto outGrad_flt = outGrad.flat<float>();
        auto inGrad_flt  = inGrad.flat<float>();
        auto stale_flt   = stale.flat<float>();

        std::cout << "XXXXX" << std::endl;

        bool t = true;
        bool f = false;
        // TODO the bug is here. check if a tensor is empty
        // compare exc does not work, 0 shape 
        if (hasInGrad.compare_exchange_strong(t, f)) {
            std::lock_guard<std::mutex> lock(inGradMutex);
            
            std::cout << inGrad.DebugString() << std::endl;

            for (int i = 0; i < grads_flt.size(); ++i) {
                 grads_flt(i) = grads_flt(i) + inGrad_flt(i);
            }
            Tensor tensorReset(DataTypeToEnum<float>::v(), gradients.shape());
            inGrad = tensorReset;
        }

        std::cout << "YYYYY" << std::endl;


        auto expire = stale.NumElements() > 0;
        // TODO: cast operations to pointers to avoid bound checks
        for (int i = 0; i < outGrad_flt.size(); ++i) {
            auto stale = expire ? stale_flt(i) : 0.0;
            outGrad_flt(i) = grads_flt(i) + outGrad_flt(i) - stale;
        }

        if (_kungfu_world.GetGlobalStep() % numberPartitions == partitionIndex) {           
            std::function<void()> func = [&]() {
                std::lock_guard<std::mutex> lock(inGradMutex);
                hasInGrad = true;
                std::cout << "I am calling the callback" << std::endl;
                
                // subract gradients from inGrad to not apply them twice
                for (int i = 0; i < inGrad_flt.size(); ++i) {
                     inGrad_flt(i) = inGrad_flt(i) - grads_flt(i);
                }
                done();
            };
            CallbackWrapper accumulatedGradientCallback(func);
             

             std::cout << "The pointer is " << (&inGrad) << std::endl;
            _kungfu_world.AllReduce(outGrad.tensor_data().data(),
                                    (void *)((&inGrad)->tensor_data().data()),
                                    outGrad.NumElements(),
                                    to_kungfu_type(outGrad.dtype()), KungFu_SUM,
                                    name().c_str(), accumulatedGradientCallback);
            // The name will not be unique in the async case because it would not be 
            // able to differentiated the traffic of the allreduce with diff global steps
            std::cout << "Before 1" << std::endl;
            *output = gradients;
            std::cout << "After 1" << std::endl;
        } else {
            std::cout << "Before 2" << std::endl;
            *output = gradients;
            std::cout << "After 2" << std::endl;
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
