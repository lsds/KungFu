
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("KungfuRequest")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("tensor_name: string")
    .Input("target: int32")
    .Input("example: T")  // FIXME: don't depend on input
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(1, c->input(1));
        return Status::OK();
    });

class Request : public AsyncOpKernel
{
    std::string name_;

  public:
    explicit Request(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &name_));
        OP_REQUIRES(context, name_.size() >= 0,
                    errors::InvalidArgument("tensor_name must not be empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int32_t target  = context->input(0).scalar<int32_t>()();
        const Tensor &example = context->input(1);
        Tensor *output        = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, example.shape(), &output),
            done);
        _kungfu_world->Request(target, name_.c_str(),
                               const_cast<char *>(output->tensor_data().data()),
                               output->NumElements(),
                               to_kungfu_type(output->dtype()), done);
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuRequest").Device(DEVICE_CPU), Request);

}  // namespace tensorflow
