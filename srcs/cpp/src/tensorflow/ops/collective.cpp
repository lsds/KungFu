#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
// The AllReduce operator takes a single tensor (e.g. the computed gradient),
// and reduce (by taking sum) with the peers, and finally returns a tensor with
// exactly the same shape.
REGISTER_OP("AllReduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

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
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world->AllReduce(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduce").Device(DEVICE_CPU), AllReduce);

REGISTER_OP("Broadcast")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class Broadcast : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world->Broadcast(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), name().c_str(),
            done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), Broadcast);

REGISTER_OP("GradientNoise")
    .Attr("input_tensor_name: string")
    .Attr("alpha: float")
    .Input("g_biased: float32")
    .Input("s_biased: float32")
    .Output("output: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class GradientNoise : public OpKernel
{
    using OpKernel::OpKernel;

public:
    std::string input_tensor_name_;
    float alpha_;
    float g_ema;
    float s_ema;

    explicit GradientNoise(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        
        OP_REQUIRES_OK(context, context->GetAttr("alpha",
                                                 &alpha_));
        OP_REQUIRES(
            context, alpha_ > 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));

        g_ema = 0.0;
        s_ema = 0.0;

        std::cout << "Constructing global var op. alpha = " << alpha_ << std::endl;
    }
    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(2, context->num_inputs());

        Tensor &g_biased_tensor = (Tensor &)context->input(0);
        Tensor &s_biased_tensor = (Tensor &)context->input(1);

        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, g_biased_tensor.shape(), &output));

        float g_current = (float) g_biased_tensor.scalar<float>()();
        float s_current = (float) s_biased_tensor.scalar<float>()();

        if (g_ema == 0.0) {
            g_ema = g_current;
        } else {
            g_ema = alpha_ * g_current + (1 - alpha_) * g_ema;
        }

        if (s_ema == 0.0) {
            s_ema = s_current;
        } else {
            s_ema = alpha_ * s_current + (1 - alpha_) * s_ema;
        }
        float gradient_noise = s_ema / g_ema;

        float *y = static_cast<float *>((void *)output->tensor_data().data());
        y[0] = gradient_noise;
    }
};

REGISTER_KERNEL_BUILDER(Name("GradientNoise").Device(DEVICE_CPU),
                        GradientNoise);


REGISTER_OP("Controller")
    .Input("negotiated_gradients: float32")
    .Output("output: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class Controller : public OpKernel
{
    using OpKernel::OpKernel;

public:

    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &negotiated_gradients = (Tensor &)context->input(0);

        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, negotiated_gradients.shape(), &output));

        *output = negotiated_gradients;
    }
};

REGISTER_KERNEL_BUILDER(Name("Controller").Device(DEVICE_CPU),
                        Controller);

}  // namespace tensorflow
