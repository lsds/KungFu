#include <kungfu/tensorflow/ops.h>
#include <tensorflow/stream_executor/stream.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(StartNcclScheduler).Input("input: string");

class StartNcclScheduler : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        const auto t_names  = input.vec<std::string>();
        std::vector<std::string> names;
        for (int i = 0; i < t_names.size(); ++i) {
            names.push_back(t_names(i));
        }
        kungfu::tensorflow::_world_gpu->StartGroup(names);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);

REGISTER_KUNGFU_OP(NcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class NcclAllReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;

  public:
    explicit NcclAllReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);

        kungfu::tensorflow::_world_gpu->AllReduce(
            [stream = context->op_device_context()->stream()]() {
                stream->BlockHostUntilDone();
            },
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(NcclAllReduce, DEVICE_GPU);
}  // namespace tensorflow
