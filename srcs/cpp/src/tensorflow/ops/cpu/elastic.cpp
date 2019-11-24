#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(StepBasedSchedule)
    .Attr("config: string")
    .Input("step: int32")
    .Output("cluster_size: int32");

class StepBasedSchedule : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    StepBasedSchedule(OpKernelConstruction *context) : OpKernel(context)
    {
        std::string config;
        OP_REQUIRES_OK(context, context->GetAttr("config", &config));
        OP_REQUIRES(context, config.size() > 0,
                    errors::InvalidArgument("config can't be empty"));
        // TODO: parse config
    }

    void Compute(OpKernelContext *context) override
    {
        const int32_t step   = context->input(0).scalar<int32_t>()();
        Tensor *cluster_size = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, MakeTensorShape(),
                                                         &cluster_size));
        cluster_size->scalar<int32_t>()() = step + 1 - step;  // TODO:
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StepBasedSchedule, DEVICE_CPU);
}  // namespace tensorflow
