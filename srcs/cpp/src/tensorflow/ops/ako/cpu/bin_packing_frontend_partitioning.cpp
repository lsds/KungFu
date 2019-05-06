#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>

namespace tensorflow
{
REGISTER_OP("PartialNegotiatorFrontEndPartitioning")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("index: int")
    .Attr("partitions: int")
    .Input("allgradients: T")
    .Output("output: T")
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
    int32_t partitions_;

    explicit PartialNegotiatorFrontEndPartitioning(
        OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("index", &index_));
        OP_REQUIRES(context, index_ >= 0,
                    errors::InvalidArgument("invalid partition index"));

        OP_REQUIRES_OK(context, context->GetAttr("partitions", &partitions_));
        OP_REQUIRES(context, partitions_ > 0,
                    errors::InvalidArgument("invalid number of partitions"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradients = (Tensor &)context->input(0);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        if (_kungfu_world->GetGlobalStep() % partitions_ == index_) {
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

REGISTER_KERNEL_BUILDER(
    Name("PartialNegotiatorFrontEndPartitioning").Device(DEVICE_CPU),
    PartialNegotiatorFrontEndPartitioning);

}  // namespace tensorflow
