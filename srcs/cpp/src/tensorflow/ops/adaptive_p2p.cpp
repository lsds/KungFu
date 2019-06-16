#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{

REGISTER_OP("AdaptiveRequestVariables")
    .Attr("T: {float32}")
    .Attr("shapes: list(shape)")
    .Attr("dtype: type")  // FIXME: infer dtype from T
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AdaptiveRequestVariables : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

    int num_tensors_;
    std::vector<TensorShapeProto> shapes_;
    DataType dtype_;

  public:
    explicit AdaptiveRequestVariables(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &num_tensors_));
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        std::vector<Tensor *> outputs(num_tensors_);
        for (int i = 0; i < num_tensors_; i++) {
            OP_REQUIRES_OK_ASYNC(
                context, context->allocate_output(i, shapes_[i], &outputs[i]),
                done);
        }
        LOG(WARNING) << "TODO : AdaptiveRequestVariables::ComputeAsync";
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("AdaptiveRequestVariables").Device(DEVICE_CPU),
                        AdaptiveRequestVariables);

}  // namespace tensorflow
