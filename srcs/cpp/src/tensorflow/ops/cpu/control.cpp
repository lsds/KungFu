#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(ResizeCluster)
    .Input("checkpoint: string")
    .Input("new_cluster_size: int32")
    // indicats if self is still in the new cluster
    .Output("keep: bool");

class ResizeCluster : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const std::string &chpt = context->input(0).scalar<std::string>()();
        const int32_t new_size  = context->input(1).scalar<int32_t>()();
        Tensor *keep            = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &keep));
        _kungfu_world->ResizeCluster(chpt.c_str(), new_size,
                                     keep->scalar<bool>().data());
        done();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ResizeCluster, DEVICE_CPU);
}  // namespace tensorflow
