#include <kungfu/tensorflow/ops.h>
#include <kungfu/utils/trace.hpp>
#include <tensorflow/stream_executor/stream.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(ScheduledNcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledNcclAllReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;

  public:
    explicit ScheduledNcclAllReduce(OpKernelConstruction *context)
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

        kungfu::tensorflow::_world_gpu->ScheduledAllReduce(
            [stream = context->op_device_context()->stream()]() {
                stream->BlockHostUntilDone();
            },
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledNcclAllReduce, DEVICE_GPU);

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

        {
            TRACE_SCOPE("BlockHostUntilDone");
            context->op_device_context()->stream()->BlockHostUntilDone();
        }

        kungfu::tensorflow::_world_gpu->AllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(NcclAllReduce, DEVICE_GPU);
}  // namespace tensorflow
