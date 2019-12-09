#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(Counter)
    .Attr("init: int = 0")
    .Attr("incr: int = 1")
    .Output("count: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class Counter : public OpKernel
{
    bool debug_;
    int32_t counter_;
    int32_t incr_;

  public:
    explicit Counter(OpKernelConstruction *context)
        : OpKernel(context), counter_(0)
    {
        context->GetAttr("debug", &debug_);
        OP_REQUIRES_OK(context, context->GetAttr("init", &counter_));
        OP_REQUIRES_OK(context, context->GetAttr("incr", &incr_));
    }

    void Compute(OpKernelContext *context) override
    {
        if (debug_) {
            LOG(WARNING) << "Counter::Compute, counter= " << counter_;
        }
        Tensor *count = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &count));
        count->scalar<int32_t>()() = counter_;
        counter_ += incr_;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Counter, DEVICE_CPU);
}  // namespace tensorflow
