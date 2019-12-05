#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(Consensus)
    .Attr("strong: bool = true")
    .Attr("tensor_name: string")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T");

class Consensus : public OpKernel
{
    std::string tensor_name_;

  public:
    explicit Consensus(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
        OP_REQUIRES(context, !tensor_name_.empty(),
                    errors::InvalidArgument("tensor_name is empty"));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        bool ok;
        _kungfu_world->Consensus(
            input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()), &ok, tensor_name_.c_str());
        if (ok) {
            LOG(ERROR) << "Consensus check OK for " << tensor_name_;
        } else {
            LOG(ERROR) << "Consensus check FAILED for " << tensor_name_;
        }
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Consensus, DEVICE_CPU);
}  // namespace tensorflow
