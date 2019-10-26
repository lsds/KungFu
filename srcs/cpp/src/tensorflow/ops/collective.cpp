#include <kungfu/tensorflow/ops.h>

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

REGISTER_OP("NoiseScale")
    .Attr("alpha: float")
    .Input("g_biased: float32")
    .Input("s_biased: float32")
    .Output("output: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class NoiseScale : public OpKernel
{
    using OpKernel::OpKernel;

    float alpha_;
    float g_ema_;
    float s_ema_;

    bool init_;

  public:
    explicit NoiseScale(OpKernelConstruction *context)
        : OpKernel(context), g_ema_(0), s_ema_(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
        OP_REQUIRES(context, alpha_ > 0,
                    errors::InvalidArgument("alpha must be greater than zero"));
        init_ = false;
    }

    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(2, context->num_inputs());

        const Tensor &g_biased_tensor = context->input(0);
        const Tensor &s_biased_tensor = context->input(1);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, g_biased_tensor.shape(), &output));

        float g_current = g_biased_tensor.scalar<float>()();
        float s_current = s_biased_tensor.scalar<float>()();

        if (!init_) {
            g_ema_ = g_current;
            s_ema_ = s_current;
            init_  = true;
        } else {
            g_ema_ = alpha_ * g_current + (1 - alpha_) * g_ema_;
            s_ema_ = alpha_ * s_current + (1 - alpha_) * s_ema_;
        }

        float noise_scale = s_ema_ / g_ema_;

        float *y = const_cast<float *>(output->scalar<float>().data());
        y[0]     = noise_scale;
    }
};

REGISTER_KERNEL_BUILDER(Name("NoiseScale").Device(DEVICE_CPU), NoiseScale);

}  // namespace tensorflow
