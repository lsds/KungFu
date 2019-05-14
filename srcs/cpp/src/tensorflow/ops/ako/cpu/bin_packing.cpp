#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>


#include <bin_packing_helpers.hpp> 

namespace tensorflow
{
REGISTER_OP("PartialNegotiator")
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

class PartialNegotiator : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;
    using CPUDevice = Eigen::ThreadPoolDevice;

  public:
    std::string input_tensor_name_;
    int32_t tensorSize_;
    int32_t count_gradients_;
    int32_t budget;

    int32_t global_step;

    explicit PartialNegotiator(OpKernelConstruction *context)
        : AsyncOpKernel(context), global_step(0)
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
        global_step++;
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradients = (Tensor &)context->input(0);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        bin_packing_simple(_partial_exchange_manager, _kungfu_world, input_tensor_name_,
    }
};

REGISTER_KERNEL_BUILDER(Name("PartialNegotiator").Device(DEVICE_CPU),
                        PartialNegotiator);

}  // namespace tensorflow
