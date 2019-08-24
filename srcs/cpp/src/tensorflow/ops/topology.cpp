#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu/mst.hpp>
#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("KungfuGetStartStep")
    .Input("version: int32")
    .Output("global_step: int64")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class GetStartStep : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        Tensor *global_step = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(0, MakeTensorShape(), &global_step), done);
        const int32_t version = context->input(0).scalar<int32_t>()();
        global_step->scalar<int64_t>()() = _kungfu_world->StartStep(version);
        done();
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuGetStartStep").Device(DEVICE_CPU),
                        GetStartStep);

REGISTER_OP("KungfuGetPeerInfo")
    .Input("version: int32")
    .Output("rank: int32")
    .Output("cluster_size: int32");

class GetPeerInfo : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        Tensor *rank         = nullptr;
        Tensor *cluster_size = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, MakeTensorShape(), &rank),
            done);
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(1, MakeTensorShape(), &cluster_size),
            done);
        const int32_t version     = context->input(0).scalar<int32_t>()();
        rank->scalar<int32_t>()() = _kungfu_world->Rank(version);
        cluster_size->scalar<int32_t>()() = _kungfu_world->ClusterSize(version);
        done();
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuGetPeerInfo").Device(DEVICE_CPU),
                        GetPeerInfo);

REGISTER_OP("KungfuGetPeerLatencies")
    .Attr("cluster_size: int")
    .Input("local_step: int64")  // FIXME: don't require input. Operator without
                                 // input but with output will be cached by TF.
    .Output("latencies: float32");

class GetPeerLatencies : public AsyncOpKernel
{
    int size_;

  public:
    explicit GetPeerLatencies(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("cluster_size", &size_));
        OP_REQUIRES(context, size_ > 0,
                    errors::InvalidArgument("size > 0 is required"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // const int64_t local_step = context->input(0).scalar<int64_t>()();
        // LOG(WARNING) << "GetPeerLatencies called with " << local_step;
        Tensor *latencies = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(0, MakeTensorShape(size_), &latencies),
            done);
        _kungfu_world->GetPeerLatencies(
            const_cast<float *>(latencies->vec<float>().data()),
            latencies->NumElements());
        done();
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuGetPeerLatencies").Device(DEVICE_CPU),
                        GetPeerLatencies);

REGISTER_OP("KungfuMinimumSpanningTree")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("weight: T")
    .Output("edges: int32");

class MinimumSpanningTree : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &weights = context->input(0);
        const int n           = weights.NumElements();
        Tensor *edges         = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(0, MakeTensorShape(n - 1, 2), &edges),
            done);
        using Weight = float;  // FIXME: use type switch on T
        _kungfu_world->AllGatherTransform(
            weights.vec<Weight>().data(), n,  //
            const_cast<int32_t *>(edges->matrix<int32_t>().data()),
            2 * (n - 1),  //
            "mst",
            [n = n](const Weight *w, int /* n*n */, int32_t *v,
                    int /* 2(n-1) */) {
                kungfu::MinimumSpanningTree<Weight, int32_t> mst;
                mst(n, w, v);
            });
        done();
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuMinimumSpanningTree").Device(DEVICE_CPU),
                        MinimumSpanningTree);

REGISTER_OP("KungfuGetNeighbourMask")
    .Attr("T: {int32}")
    .Attr("cluster_size: int")  // TODO: make it an Input
    .Attr("self_rank: int")     // TODO: make it an Input
    .Input("edges: T")
    .Output("mask: bool");

class GetNeighbourMask : public OpKernel
{
    int size_;
    int self_;

  public:
    explicit GetNeighbourMask(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("cluster_size", &size_));
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_));
        OP_REQUIRES(context, 0 <= self_ && self_ < size_,
                    errors::InvalidArgument(
                        "self_rank in [0, cluster_size) is required"));
    }

    void Compute(OpKernelContext *context) override
    {
        const auto edges = context->input(0).matrix<int32_t>();
        Tensor *mask     = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, MakeTensorShape(size_), &mask));
        bool *m = const_cast<bool *>(mask->vec<bool>().data());
        std::fill(m, m + size_, false);
        // FIXME: check shape of edges is [size_ - 1, 2]
        for (int i = 0; i < size_ - 1; ++i) {
            const int u = edges(i, 0);
            const int v = edges(i, 1);
            if (u == self_) { m[v] = true; }
            if (v == self_) { m[u] = true; }
        }
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuGetNeighbourMask").Device(DEVICE_CPU),
                        GetNeighbourMask);

REGISTER_OP("KungfuRoundRobin").Input("mask: bool").Output("choice: int32");

class RoundRobin : public OpKernel
{
    int pos_;

  public:
    explicit RoundRobin(OpKernelConstruction *context)
        : OpKernel(context), pos_(0)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        const auto mask = context->input(0);
        const auto m    = mask.vec<bool>();
        Tensor *choice  = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &choice));
        const int n = mask.NumElements();
        auto y      = choice->scalar<int32_t>();
        for (int i = 0; i < n; ++i) {
            const int idx = (pos_ + i) % n;
            if (m(idx)) {
                pos_ = (idx + 1) % n;
                y()  = idx;
                return;
            }
        }
        LOG(WARNING) << "no choice available from mask";
        y() = -1;  // failed
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuRoundRobin").Device(DEVICE_CPU),
                        RoundRobin);

}  // namespace tensorflow
