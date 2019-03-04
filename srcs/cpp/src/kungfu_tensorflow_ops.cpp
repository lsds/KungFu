#include <tensorflow/core/framework/op_kernel.h>

#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_ops.h>
#include <queue>
#include <mutex>
#include <atomic>

#include <chrono>

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <csignal>

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
    using CPUDevice = Eigen::ThreadPoolDevice;

  public:
    // have them public for now

    std::queue<Tensor> tensorWindow;
    Tensor outGrad; // the accumulated gradient to be negotiated
    Tensor inGrad;  // the accumulated gradient received through negotiation
    std::mutex allMutex; // protects
    bool hasInGrad;
    int id;
    bool isInit;

    explicit AkoNegotiator(OpKernelConstruction* context) : AsyncOpKernel(context) {
        hasInGrad = false;
        id = std::rand();
        isInit = false;
    }

    // creates a TF pool to  perform the operation
    // bool IsExpensive() override { return true; }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // Check arg count: gradient tensors group, number partitions, current
        // partition index
        DCHECK_EQ(4, context->num_inputs());

        
        Tensor &gradients                   = (Tensor&) context->input(0);    

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

        std::lock_guard<std::mutex> lock(allMutex);


        std::cout << gradients.DebugString() << std::endl;
        std::cout << outGrad.DebugString() << std::endl;
        std::cout << inGrad.DebugString() << std::endl;



        // Update gradient window
        Tensor stale;
        tensorWindow.push(gradients);    
        if(tensorWindow.size() > numberPartitions) {
            stale = tensorWindow.front();
            tensorWindow.pop(); 
        }
        if(!isInit) {
            Tensor zeros(DataTypeToEnum<float>::v(), gradients.shape());
            for (int i = 0; i < zeros.flat<float>().size(); ++i) {
                zeros.flat<float>()(i) = 0.0;
            }
            outGrad = zeros;
            inGrad = zeros;

            // std::cout << outGrad.DebugString() << std::endl;
            // std::cout << inGrad.DebugString() << std::endl;

            // auto inGrad_flt = inGrad.flat<float>();
            // inGrad_flt.setZero();
            // auto outGrad_flt = outGrad.flat<float>();
            // outGrad_flt.setZero();
            isInit = true;
        }


        // Create snapshots right before you use the tensors
        if (hasInGrad) {
            auto grads_flt   = gradients.flat<float>();
            auto inGrad_flt  = inGrad.flat<float>();

            DCHECK_EQ(grads_flt.size(), inGrad_flt.size());
            grads_flt = grads_flt + inGrad_flt;
            hasInGrad = false;
        } 


        int inGrad_size = static_cast<int>(inGrad.NumElements());
        float *inGrad_ptr = inGrad.flat<float>().data();
        for(int i = 0; i < inGrad_size; i++) {
            inGrad_ptr[i] = 0.0;
        }


        auto inGrad_flt = inGrad.flat<float>();
        auto grads_flt   = gradients.flat<float>();
        auto outGrad_flt = outGrad.flat<float>();
        auto stale_flt   = stale.flat<float>();
        auto expire = stale.NumElements() > 0;

        DCHECK_EQ(outGrad_flt.size(), grads_flt.size());
        if(expire) {
            DCHECK_EQ(outGrad_flt.size(), stale_flt.size());
            // std::cout << "gradients: " << gradients.DebugString() << std::endl;
            // std::cout << "stale: " << stale.DebugString() << std::endl;
            // std::cout << "outGrad before: " << outGrad.DebugString() << std::endl;            
            // int size = static_cast<int>(outGrad.NumElements());
            // float *grads_ptr   = grads_flt.data();
            // float *outGrad_ptr = outGrad_flt.data();
            // float *stale_ptr   = stale_flt.data();
            // for(int i = 0; i < size; i++) {
            //     outGrad_ptr[i] = static_cast<float>((double)grads_ptr[i] + (double)outGrad_ptr[i] - (double)stale_ptr[i]); 
            // } 
           
            outGrad_flt = grads_flt + outGrad_flt - stale_flt;
            //DCHECK_EQ(outGrad_flt, grads_flt + outGrad_flt - stale_flt);
            // TODO: cast operations to pointers to avoid bound checks

            // functor::ApplyAccumulate<Device, float>()(
            //      device, outGrad.flat<float>(), gradients.flat<float>(), stale.flat<float>());            
            // outGrad_flt.device(d) = grads_flt + outGrad_flt - stale_flt;

            // std::cout << outGrad_ptr[0] << std::endl;
            // std::cout << "outGrad after: " << outGrad.DebugString() << std::endl;            
            // for (int i = 0; i < size; ++i) {
            //     DCHECK_EQ(outGrad_ptr[i], grads_ptr[i] + outGrad_ptr[i] - stale_ptr[i]);
            // }
            
           
            // Check this if case is correct
            // if subtraction is correctly performed, this should not happen
            // understand the adam way and the benefits it provides (more fine grained locking?)
            // checkpoint meeting on monday
        } else {
            // for (int i = 0; i < outGrad_flt.size(); ++i) {
            //     outGrad_flt(i) = grads_flt(i) + outGrad_flt(i);
            // }
            outGrad_flt = grads_flt + outGrad_flt;
            // float *grads_ptr   = grads_flt.data();
            // float *outGrad_ptr = outGrad_flt.data();
            // int size = static_cast<int>(outGrad.NumElements());
            // for(int i = 0; i < size; i++) {
            //     outGrad_ptr[i] = static_cast<float>((double)grads_ptr[i] + (double)outGrad_ptr[i]);
            // } 
        }
        

        if (_kungfu_world.GetGlobalStep() % numberPartitions == partitionIndex) {           
            std::function<void()> func = [&, done]() {
                std::lock_guard<std::mutex> l(allMutex);
                
                auto grads_flt_cb   = gradients.flat<float>();
                auto inGrad_flt_cb  = inGrad.flat<float>();
                float *grads_ptr    = grads_flt_cb.data();
                float *inGrad_ptr   = inGrad_flt_cb.data();
                int   size = static_cast<int>(inGrad.NumElements());
                
                // subract gradients from inGrad to not apply them twice
                DCHECK_EQ(inGrad_flt_cb.size(), grads_flt_cb.size());
                // for(int i = 0; i < size; i++) {
                //     inGrad_ptr[i] = static_cast<float>((double)inGrad_ptr[i] - (double)grads_ptr[i]);
                // }
                inGrad_flt_cb = inGrad_flt_cb - grads_flt_cb;

                hasInGrad = true;
                done();
            };
            _kungfu_world.AllReduce(outGrad.tensor_data().data(),
                                    (void *)(inGrad.tensor_data().data()),
                                    outGrad.NumElements(),
                                    to_kungfu_type(outGrad.dtype()), KungFu_SUM,
                                    name().c_str(), func);
            // The name will not be unique in the async case because it would not be 
            // able to differentiated the traffic of the allreduce with diff global steps
            *output = gradients;
        } else {
            *output = gradients;
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
