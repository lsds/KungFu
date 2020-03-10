#include <kungfu/tensorflow/ops.h>

#include <kungfu/utils/trace.hpp>

namespace tensorflow
{
REGISTER_KUNGFU_OP(Barrier);

class Barrier : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        _kungfu_world->Barrier(done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Barrier, DEVICE_CPU);

REGISTER_KUNGFU_OP(Consensus)
    .Attr("strong: bool = true")  // TODO: support weak check
    .Attr("tensor_name: string")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: bool");

class Consensus : public AsyncOpKernel
{
    std::string tensor_name_;

  public:
    explicit Consensus(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
        OP_REQUIRES(context, !tensor_name_.empty(),
                    errors::InvalidArgument("tensor_name is empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, MakeTensorShape(), &output),
            done);
        _kungfu_world->Consensus(
            input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()),
            reinterpret_cast<bool *>(output->scalar<bool>().data()),
            tensor_name_.c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Consensus, DEVICE_CPU);

// The AllReduce operator takes a single tensor (e.g. the computed gradient),
// and reduce (by taking sum) with the peers, and finally returns a tensor with
// exactly the same shape.
REGISTER_KUNGFU_OP(AllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("op: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AllReduce : public AsyncOpKernel
{
    KungFu_Op op_;

  public:
    explicit AllReduce(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
        std::string op;
        OP_REQUIRES_OK(context, context->GetAttr("op", &op));
        static const std::map<std::string, KungFu_Op> kungfu_op({
            {"sum", KungFu_SUM},
            {"min", KungFu_MIN},
            {"max", KungFu_MAX},
            {"prod", KungFu_PROD},
        });
        OP_REQUIRES(context, kungfu_op.count(op) > 0,
                    errors::InvalidArgument("invalid op"));
        op_ = kungfu_op.at(op);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        _kungfu_world->AllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), op_,
            name().c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(AllReduce, DEVICE_CPU);

// The SpotnikAllReduce operator takes a single tensor (e.g. the computed gradient),
// and reduce (by taking sum) with the peers, and finally returns a tensor with
// exactly the same shape and if the operation succeeded.
REGISTER_KUNGFU_OP(SpotnikAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("op: string")
    .Input("input: T")
    .Output("output: T")
    .Output("succeeded: int32")
    .SetShapeFn(shape_inference::UnchangedShape);

class SpotnikAllReduce : public AsyncOpKernel
{
    KungFu_Op op_;

  public:
    explicit SpotnikAllReduce(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
        std::string op;
        OP_REQUIRES_OK(context, context->GetAttr("op", &op));
        static const std::map<std::string, KungFu_Op> kungfu_op({
            {"sum", KungFu_SUM},
            {"min", KungFu_MIN},
            {"max", KungFu_MAX},
            {"prod", KungFu_PROD},
        });
        OP_REQUIRES(context, kungfu_op.count(op) > 0,
                    errors::InvalidArgument("invalid op"));
        op_ = kungfu_op.at(op);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        Tensor *succeeded      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(1, MakeTensorShape(), &succeeded), done);
        _kungfu_world->SpotnikAllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()),
            const_cast<int32_t *>(succeeded->scalar<int32_t>().data()),
            op_,
            name().c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(SpotnikAllReduce, DEVICE_CPU);

REGISTER_KUNGFU_OP(Broadcast)
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

REGISTER_KUNGFU_KERNEL_BUILDER(Broadcast, DEVICE_CPU);

REGISTER_KUNGFU_OP(NoiseScale)
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

REGISTER_KUNGFU_KERNEL_BUILDER(NoiseScale, DEVICE_CPU);
}  // namespace tensorflow
