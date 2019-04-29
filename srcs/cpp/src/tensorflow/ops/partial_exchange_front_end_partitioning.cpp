#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>

namespace tensorflow
{
REGISTER_OP("PartialNegotiatorFrontEndPartitioning")
    .Attr("input_tensor_name: string")
    .Attr("budget: int")
    .Attr("tensor_size: int")
    .Attr("count_gradients: int")
    .Input("allgradients: float32")
    .Output("output: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class PartialNegotiatorFrontEndPartitioning : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;
    using CPUDevice = Eigen::ThreadPoolDevice;

  public:
    int32_t index_;

    explicit PartialNegotiatorFrontEndPartitioning(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("index",
                                                 &index_));
        OP_REQUIRES(
            context, index_ >= 0,
            errors::InvalidArgument("invalid partition index"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradients = (Tensor &)context->input(0);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        if (_kungfu_world->GetGlobalStep() % (index + 1) == 0) {
            _kungfu_world->AllReduce(gradients.tensor_data().data(),
                                     (void *)(output->tensor_data().data()),
                                     gradients.NumElements(),
                                     to_kungfu_type(gradients.dtype()),
                                     KungFu_SUM, name().c_str(), done);
            // Because it is synchronous, the done callback will signal when the
            // value held
            // in the memory where output points to is ready to be used.
        } else {
            *output = gradients;
            done();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PartialNegotiatorFrontEndPartitioning").Device(DEVICE_CPU),
                        PartialNegotiatorFrontEndPartitioning);

}  // namespace tensorflow
