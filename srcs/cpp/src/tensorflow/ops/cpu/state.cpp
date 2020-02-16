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
        OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
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

REGISTER_KUNGFU_OP(ExponentialMovingAverage)
    .Attr("alpha: float")
    .Attr("T: {float32}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ExponentialMovingAverage : public OpKernel
{
    float alpha_;
    float count_;
    float value_;

  public:
    explicit ExponentialMovingAverage(OpKernelConstruction *context)
        : OpKernel(context), count_(0), value_(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    }

    void Compute(OpKernelContext *context) override
    {
        const float x  = context->input(0).scalar<float>()();
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        if (count_++ == 0) {
            value_ = x;
        } else {
            value_ = alpha_ * value_ + (1 - alpha_) * x;
        }
        output->scalar<float>()() = value_;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ExponentialMovingAverage, DEVICE_CPU);
}  // namespace tensorflow
