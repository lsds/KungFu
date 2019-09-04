
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("KungfuRequestVariable")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("tensor_name: string")
    .Attr("shape: shape")
    .Attr("use_version: bool")
    .Input("target: int32")
    .Input("version: int64")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        shape_inference::ShapeHandle handle;
        TensorShapeProto shape;
        c->GetAttr("shape", &shape);
        c->MakeShapeFromShapeProto(shape, &handle);
        c->set_output(0, handle);
        return Status::OK();
    });

class RequestVariable : public AsyncOpKernel
{
    std::string tensor_name_;
    TensorShapeProto shape_;
    bool use_version_;

  public:
    explicit RequestVariable(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
        OP_REQUIRES(context, tensor_name_.size() >= 0,
                    errors::InvalidArgument("tensor_name must not be empty"));
        OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
        OP_REQUIRES_OK(context, context->GetAttr("use_version", &use_version_));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int32_t target  = context->input(0).scalar<int32_t>()();
        const int64_t version = context->input(1).scalar<int64_t>()();
        Tensor *output        = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, shape_, &output), done);
        if (use_version_) {
            _kungfu_world->Request(
                target, std::to_string(version).c_str(), tensor_name_.c_str(),
                const_cast<char *>(output->tensor_data().data()),
                output->NumElements(), to_kungfu_type(output->dtype()), done);
        } else {
            _kungfu_world->Request(
                target, tensor_name_.c_str(),
                const_cast<char *>(output->tensor_data().data()),
                output->NumElements(), to_kungfu_type(output->dtype()), done);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("KungfuRequestVariable").Device(DEVICE_CPU),
                        RequestVariable);
}  // namespace tensorflow
