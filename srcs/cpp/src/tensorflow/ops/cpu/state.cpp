#include <kungfu/tensorflow/ops.h>
#include <kungfu/utils/ema.hpp>

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
    using T     = float;
    using ema_t = kungfu::ExponentialMovingAverage<T>;
    std::unique_ptr<ema_t> ema_;

  public:
    explicit ExponentialMovingAverage(OpKernelConstruction *context)
        : OpKernel(context)
    {
        T alpha;
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
        ema_.reset(new ema_t(alpha));
    }

    void Compute(OpKernelContext *context) override
    {
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        output->scalar<T>()() = ema_->update(context->input(0).scalar<T>()());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ExponentialMovingAverage, DEVICE_CPU);
}  // namespace tensorflow
