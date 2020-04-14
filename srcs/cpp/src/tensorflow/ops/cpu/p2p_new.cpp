#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(RequestVariable)
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
        TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape, &handle));
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
            _default_peer->Request(
                target, std::to_string(version).c_str(), tensor_name_.c_str(),
                const_cast<char *>(output->tensor_data().data()),
                output->NumElements(), to_kungfu_type(output->dtype()), done);
        } else {
            _default_peer->Request(
                target, tensor_name_.c_str(),
                const_cast<char *>(output->tensor_data().data()),
                output->NumElements(), to_kungfu_type(output->dtype()), done);
        }
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(RequestVariable, DEVICE_CPU);
}  // namespace tensorflow
