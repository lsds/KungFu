#include <kungfu/ncclv2/helper.hpp>
#include <kungfu/tensorflow/gpu_helpers.hpp>

namespace tensorflow
{
REGISTER_KUNGFU_OP(ScheduledNcclAllReduceV2)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("op_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledNcclAllReduceV2 : public AsyncOpKernel
{
    std::string op_name_;

  public:
    explicit ScheduledNcclAllReduceV2(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("op_name", &op_name_));
        OP_REQUIRES(context, op_name_.size() >= 0,
                    errors::InvalidArgument("op_name must not be empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        const auto w     = make_workspace(input, output);
        auto ready_event = create_init_ready_event(context);
        _default_nccl_helper_v2->ScheduleAllReduce(
            w, [=] { wait_delete_ready_event(ready_event); }, op_name_, done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledNcclAllReduceV2, DEVICE_GPU);
}  // namespace tensorflow
