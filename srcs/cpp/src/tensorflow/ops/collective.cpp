#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu/ema.hpp>
#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("KungfuBarrier");

class KungfuBarrier : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        _kungfu_world->Barrier(done);
    }
};

REGISTER_KERNEL_BUILDER(Name("KungfuBarrier").Device(DEVICE_CPU),
                        KungfuBarrier);

// The AllReduce operator takes a single tensor (e.g. the computed gradient),
// and reduce (by taking sum) with the peers, and finally returns a tensor with
// exactly the same shape.
REGISTER_OP("AllReduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AllReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;

  public:
    explicit AllReduce(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
    }

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        _kungfu_world->AllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduce").Device(DEVICE_CPU), AllReduce);

REGISTER_OP("Broadcast")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class Broadcast : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        _kungfu_world->Broadcast(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), name().c_str(),
            done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), Broadcast);

REGISTER_OP("GradientNoise")
    .Attr("alpha: float")
    .Input("g_biased: float32")
    .Input("s_biased: float32")
    .Output("output: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class GradientNoise : public OpKernel
{
    using OpKernel::OpKernel;

    using T     = float;
    using ema_t = ExponentialMovingAverage<T>;

    T alpha_;
    std::unique_ptr<ema_t> g_ema_;
    std::unique_ptr<ema_t> s_ema_;

  public:
    explicit GradientNoise(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
        OP_REQUIRES(context, alpha_ > 0,
                    errors::InvalidArgument("alpha must be greater than zero"));
        g_ema_.reset(new ema_t(alpha_));
        s_ema_.reset(new ema_t(alpha_));
    }

    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(2, context->num_inputs());
        const T g = context->input(0).scalar<T>()();
        const T s = context->input(1).scalar<T>()();
        g_ema_->update(g);
        s_ema_->update(s);
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        output->scalar<T>()() = s_ema_->get() / g_ema_->get();
    }
};

REGISTER_KERNEL_BUILDER(Name("GradientNoise").Device(DEVICE_CPU),
                        GradientNoise);

}  // namespace tensorflow
