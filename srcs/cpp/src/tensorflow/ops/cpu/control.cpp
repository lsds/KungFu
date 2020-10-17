#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(ResizeClusterFromURL)
    // indicats if cluster is changed
    .Output("changed: bool")
    // indicats if self is detached from the old cluster
    .Output("detached: bool")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
    });

class ResizeClusterFromURL : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        Tensor *changed = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(), &changed));
        Tensor *detached = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(1, MakeTensorShape(), &detached));
        _default_peer->ResizeClusterFromURL(changed->scalar<bool>().data(),
                                            detached->scalar<bool>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResizeClusterFromURL, DEVICE_CPU);

REGISTER_KUNGFU_OP(ResizeCluster)
    .Input("new_size: uint32")
    // indicats if cluster is changed
    .Output("changed: bool")
    // indicats if self is detached from the old cluster
    .Output("detached: bool")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
    });

class ResizeCluster : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &new_size = context->input(0);
        Tensor *changed        = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(), &changed));
        Tensor *detached = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(1, MakeTensorShape(), &detached));
        _default_peer->ResizeCluster(new_size.scalar<uint32_t>()(),
                                     changed->scalar<bool>().data(),
                                     detached->scalar<bool>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResizeCluster, DEVICE_CPU);
}  // namespace tensorflow
