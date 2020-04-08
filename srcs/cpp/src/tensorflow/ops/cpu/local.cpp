#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(SaveVariable)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("use_version: bool")
    .Input("version: int64")
    .Input("input: T");

class SaveVariable : public AsyncOpKernel
{
    std::string input_tensor_name_;
    bool use_version_;

  public:
    explicit SaveVariable(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        OP_REQUIRES_OK(context, context->GetAttr("use_version", &use_version_));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int64_t version = context->input(0).scalar<int64_t>()();
        const Tensor &input   = context->input(1);
        if (use_version_) {
            _default_peer->Save(std::to_string(version).c_str(),
                                input_tensor_name_.c_str(),
                                input.tensor_data().data(), input.NumElements(),
                                to_kungfu_type(input.dtype()), done);
        } else {
            _default_peer->Save(input_tensor_name_.c_str(),
                                input.tensor_data().data(), input.NumElements(),
                                to_kungfu_type(input.dtype()), done);
        }
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(SaveVariable, DEVICE_CPU);

REGISTER_KUNGFU_OP(SaveVariables)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("names: list(string)")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T");

class SaveVariables : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

    int num_tensors_;
    std::vector<std::string> names_;

  public:
    explicit SaveVariables(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &num_tensors_));
        OP_REQUIRES_OK(context, context->GetAttr("names", &names_));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        for (int i = 0; i < num_tensors_; ++i) {
            const Tensor &t = context->input(i);
            // TODO: get name from t
            _default_peer->Save(names_.at(i).c_str(), t.tensor_data().data(),
                                t.NumElements(), to_kungfu_type(t.dtype()));
        }
        done();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(SaveVariables, DEVICE_CPU);
}  // namespace tensorflow
