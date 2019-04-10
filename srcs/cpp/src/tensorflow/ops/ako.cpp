#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <queue>
#include <mutex>

namespace tensorflow
{
REGISTER_OP("AkoNegotiator")
    .Input("allgradients: float32")
    .Input("partition: int32")
    .Input("partitioncount: int32")
    .Output("output: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
});

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
    bool isInit;

    explicit AkoNegotiator(OpKernelConstruction* context) : AsyncOpKernel(context) {
        hasInGrad = false;
        isInit = false;
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // Check arg count: gradient tensors group, number partitions, current
        // partition index
        DCHECK_EQ(3, context->num_inputs());

        
        Tensor &gradients                   = (Tensor&) context->input(0);    

        const Tensor &currentPartitionIndex = context->input(1);
        const Tensor &pAkoPartitions        = context->input(2);

        
        Tensor *output                      = nullptr;

        auto currentPartitionIndexTensor = currentPartitionIndex.vec<int>();
        auto numberPartitionsTensor      = pAkoPartitions.vec<int>();

        uint32_t numberPartitions = numberPartitionsTensor(0);
        uint32_t partitionIndex   = currentPartitionIndexTensor(0);

        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        std::lock_guard<std::mutex> lock(allMutex);

        // Update gradient window
        Tensor stale;
        tensorWindow.push(gradients);    
        if(tensorWindow.size() > numberPartitions) {
            stale = tensorWindow.front();
            tensorWindow.pop(); 
        }

        if(!isInit) {
            Tensor zeros(DataTypeToEnum<float>::v(), gradients.shape());
            auto zeros_flt = zeros.flat<float>();
            zeros_flt.setZero();
            outGrad = zeros;
            outGrad.flat<float>().setZero();
            inGrad = zeros;
            inGrad.flat<float>().setZero();
            isInit = true;
        }

        // Create snapshots right before you use the tensors
        if (hasInGrad) {
            gradients.flat<float>() = gradients.flat<float>() + inGrad.flat<float>();
            hasInGrad = false;
        } 

        // Important: reset inbound gradients
        inGrad.flat<float>().setZero();
    
        auto inGrad_flt  = inGrad.flat<float>();
        auto grads_flt   = gradients.flat<float>();
        auto outGrad_flt = outGrad.flat<float>();
        auto stale_flt   = stale.flat<float>();
        auto expire      = stale.NumElements() > 0;

        if(expire) {
            outGrad_flt = grads_flt + outGrad_flt - stale_flt;
        } else {
            outGrad_flt = grads_flt + outGrad_flt;
        }
        
        if (tensorWindow.size() > 0) {
            outGrad_flt = outGrad_flt / outGrad_flt.constant(tensorWindow.size()    );
        } else {
            std::cout << "Ako accumulation window empty!" << std::endl;
        }

        if (_kungfu_world->GetGlobalStep() % numberPartitions == partitionIndex) {           
            // Create a callback to accumulate gradients from other peers
            std::function<void()> func = [&, done]() {
                std::lock_guard<std::mutex> l(allMutex);
                
                // subract gradients from inGrad to not apply them twice
                inGrad.flat<float>() = inGrad.flat<float>() - gradients.flat<float>();
                hasInGrad = true;
                done();
            };

            _kungfu_world->AllReduce(outGrad.tensor_data().data(),
                                    (void *)(inGrad.tensor_data().data()),
                                    outGrad.NumElements(),
                                    to_kungfu_type(outGrad.dtype()), KungFu_SUM,
                                    name().c_str(), func);
            *output = gradients;
        } else {
            *output = gradients;
            done();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AkoNegotiator").Device(DEVICE_CPU),
                        AkoNegotiator);

}  // namespace tensorflow
