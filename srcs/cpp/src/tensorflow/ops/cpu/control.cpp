#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(ResizeCluster)
    .Input("new_cluster_size: int32")
    // indicats if cluster is changed
    .Output("changed: bool")
    // indicats if self is still in the new cluster
    .Output("keep: bool")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
    });

class ResizeCluster : public OpKernel
{
    bool debug_;

  public:
    ResizeCluster(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
    }

    void Compute(OpKernelContext *context) override
    {
        const int32_t new_size = context->input(0).scalar<int32_t>()();
        if (debug_) {
            LOG(WARNING) << "ResizeCluster::Compute called with chpt: "
                         << " new size: " << new_size;
        }
        Tensor *changed = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(), &changed));
        Tensor *keep = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, MakeTensorShape(), &keep));
        _kungfu_world->ResizeCluster(new_size, changed->scalar<bool>().data(),
                                     keep->scalar<bool>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResizeCluster, DEVICE_CPU);

REGISTER_KUNGFU_OP(ResizeClusterFromURL)
    // indicats if cluster is changed
    .Output("changed: bool")
    // indicats if self is still in the new cluster
    .Output("keep: bool")
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
        Tensor *keep = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, MakeTensorShape(), &keep));
        _kungfu_world->ResizeClusterFromURL(changed->scalar<bool>().data(),
                                            keep->scalar<bool>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResizeClusterFromURL, DEVICE_CPU);
}  // namespace tensorflow
