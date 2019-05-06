#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/stream_executor/stream.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>

namespace tensorflow
{
REGISTER_OP("PartialNegotiatorGpu")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("budget: int")
    .Attr("tensor_size: int")
    .Attr("count_gradients: int")
    .Input("allgradients: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class PartialNegotiatorGpu : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;
    using CPUDevice = Eigen::ThreadPoolDevice;

  public:
    std::string input_tensor_name_;
    int32_t tensorSize_;
    int32_t count_gradients_;
    int32_t budget;

    explicit PartialNegotiatorGpu(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));

        OP_REQUIRES_OK(context, context->GetAttr("budget", &budget));
        OP_REQUIRES(context, budget > 0,
                    errors::InvalidArgument("budget must be greater than 0"));

        OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &tensorSize_));
        OP_REQUIRES(
            context, tensorSize_ > 0,
            errors::InvalidArgument("tensor size must be greater than 0"));

        OP_REQUIRES_OK(context,
                       context->GetAttr("count_gradients", &count_gradients_));
        OP_REQUIRES(
            context, count_gradients_ > 0,
            errors::InvalidArgument("gradient count must be greater than 0"));

        _partial_exchange_manager->setCountGradients(count_gradients_);
        _partial_exchange_manager->setBudget(budget);
        _partial_exchange_manager->addTensorInfo(input_tensor_name_,
                                                 tensorSize_);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradients = (Tensor &)context->input(0);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        if (_partial_exchange_manager->isReadyForNegotiation(
                input_tensor_name_, _kungfu_world->GetGlobalStep())) {
            kungfu::tensorflow::_world_gpu->AllReduce(
                [stream = context->op_device_context()->stream()]() {
                    stream->BlockHostUntilDone();
                },
                gradients.tensor_data().data(),
                (void *)(output->tensor_data().data()), gradients.NumElements(),
                to_kungfu_type(gradients.dtype()), KungFu_SUM, name().c_str(),
                done);
            // Because it is synchronous, the done callback will signal when the
            // value held
            // in the memory where output points to is ready to be used.
        } else {
            *output = gradients;
            done();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PartialNegotiatorGpu").Device(DEVICE_GPU),
                        PartialNegotiatorGpu);

}  // namespace tensorflow
