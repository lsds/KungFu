#include <kungfu/tensorflow/ops.h>
#include <kungfu/utils/ema.hpp>
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
        _default_peer->Barrier(done);
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
        _default_peer->Consensus(
            input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()),
            reinterpret_cast<bool *>(output->scalar<bool>().data()),
            tensor_name_.c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Consensus, DEVICE_CPU);

static const std::map<std::string, KungFu_Op> kungfu_ops({
    {"sum", KungFu_SUM},
    {"min", KungFu_MIN},
    {"max", KungFu_MAX},
    {"prod", KungFu_PROD},
});

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
        OP_REQUIRES(context, kungfu_ops.count(op) > 0,
                    errors::InvalidArgument("invalid op"));
        op_ = kungfu_ops.at(op);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        _default_peer->AllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), op_,
            name().c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(AllReduce, DEVICE_CPU);

REGISTER_KUNGFU_OP(MonitoredAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("op: string")
    .Input("input: T")
    .Input("tree: int32")
    .Output("output: T")
    // .Output("metrics: T") // TODO: return monitoring metrics
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        // c->set_output(1, ?); // TODO(optional)
        return Status::OK();
    });

class MonitoredAllReduce : public AsyncOpKernel
{
    KungFu_Op op_;

  public:
    explicit MonitoredAllReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        std::string op;
        OP_REQUIRES_OK(context, context->GetAttr("op", &op));
        OP_REQUIRES(context, kungfu_ops.count(op) > 0,
                    errors::InvalidArgument("invalid op"));
        op_ = kungfu_ops.at(op);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        const auto &tree    = context->input(1).vec<int32_t>();
        Tensor *output      = nullptr;
        // Tensor *metrics     = nullptr; // TODO
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        // OP_REQUIRES_OK_ASYNC(
        //     context, context->allocate_output(1, input.shape(), &metrics),
        //     done);  // TODO
        _default_peer->MonitoredAllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), op_,
            tree.size() != 0 ? tree.data() : nullptr, name().c_str(), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(MonitoredAllReduce, DEVICE_CPU);

REGISTER_KUNGFU_OP(AllGather)
    .Attr("T: {int32, int64, float16, float32, float64, bool}")
    .Input("input: T")
    .Output("output: T");
// .SetShapeFn(shape_inference::UnchangedShape);

class AllGather : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        const int np        = _default_peer->Size();
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(0, BatchTensorShape(input.shape(), np),
                                     &output),
            done);
        _default_peer->AllGather(
            input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()),
            const_cast<char *>(output->tensor_data().data()), name().c_str(),
            done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(AllGather, DEVICE_CPU);

REGISTER_KUNGFU_OP(Broadcast)
    .Attr("T: {int32, int64, float16, float32, float64, bool}")
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
        _default_peer->Broadcast(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), name().c_str(),
            done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Broadcast, DEVICE_CPU);

REGISTER_KUNGFU_OP(NoiseScale)
    .Attr("alpha: float")
    .Attr("T: {float32}")
    .Input("g_biased: T")
    .Input("s_biased: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class NoiseScale : public OpKernel
{
    using T     = float;
    using ema_t = kungfu::ExponentialMovingAverage<T>;

    bool debug_;
    std::unique_ptr<ema_t> g_ema_;
    std::unique_ptr<ema_t> s_ema_;

  public:
    explicit NoiseScale(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
        if (debug_) { LOG(WARNING) << "NoiseScale created"; }
        T alpha;
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
        OP_REQUIRES(context, alpha > 0,
                    errors::InvalidArgument("alpha must be greater than zero"));
        g_ema_.reset(new ema_t(alpha));
        s_ema_.reset(new ema_t(alpha));
    }

    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(2, context->num_inputs());

        const Tensor &g_biased_tensor = context->input(0);
        const Tensor &s_biased_tensor = context->input(1);
        Tensor *output                = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, g_biased_tensor.shape(), &output));
        const T g_ema         = g_ema_->update(g_biased_tensor.scalar<T>()());
        const T s_ema         = s_ema_->update(s_biased_tensor.scalar<T>()());
        output->scalar<T>()() = s_ema / g_ema;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(NoiseScale, DEVICE_CPU);
}  // namespace tensorflow
