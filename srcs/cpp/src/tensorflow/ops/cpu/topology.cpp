#include <kungfu/mst.hpp>
#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(Rank)
    .Output("rank: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class Rank : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        Tensor *rank = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &rank));
        rank->scalar<int32_t>()() = _default_peer->Rank();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Rank, DEVICE_CPU);

REGISTER_KUNGFU_OP(ClusterSize)
    .Output("cluster_size: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class ClusterSize : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        Tensor *cluster_size = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, MakeTensorShape(),
                                                         &cluster_size));
        cluster_size->scalar<int32_t>()() = _default_peer->Size();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ClusterSize, DEVICE_CPU);

REGISTER_KUNGFU_OP(GetPeerInfo)
    .Output("rank: int32")
    .Output("cluster_size: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
    });

class GetPeerInfo : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        Tensor *rank         = nullptr;
        Tensor *cluster_size = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &rank));
        OP_REQUIRES_OK(context, context->allocate_output(1, MakeTensorShape(),
                                                         &cluster_size));
        rank->scalar<int32_t>()()         = _default_peer->Rank();
        cluster_size->scalar<int32_t>()() = _default_peer->Size();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(GetPeerInfo, DEVICE_CPU);

REGISTER_KUNGFU_OP(GetPeerLatencies)
    .Attr("cluster_size: int")
    .SetIsStateful()
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
        Tensor *latencies = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(0, MakeTensorShape(size_), &latencies),
            done);
        _default_peer->GetPeerLatencies(
            const_cast<float *>(latencies->vec<float>().data()),
            latencies->NumElements());
        done();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(GetPeerLatencies, DEVICE_CPU);

REGISTER_KUNGFU_OP(MinimumSpanningTree)
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
        _default_peer->AllGatherTransform(
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

REGISTER_KUNGFU_KERNEL_BUILDER(MinimumSpanningTree, DEVICE_CPU);

REGISTER_KUNGFU_OP(GetNeighbour)
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

REGISTER_KUNGFU_KERNEL_BUILDER(GetNeighbourMask, DEVICE_CPU);

REGISTER_KUNGFU_OP(RoundRobin).Input("mask: bool").Output("choice: int32");

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

REGISTER_KUNGFU_KERNEL_BUILDER(RoundRobin, DEVICE_CPU);
}  // namespace tensorflow
