#include <kungfu_tensorflow_ops.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>

namespace tensorflow
{
REGISTER_OP("KungfuResizeCluster")
    .Input("checkpoint: string")
    .Input("new_cluster_size: int32")
    // indicats if self is still in the new cluster
    .Output("keep: bool");

class KungfuResizeCluster : public AsyncOpKernel
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

REGISTER_KERNEL_BUILDER(Name("KungfuResizeCluster").Device(DEVICE_CPU),
                        KungfuResizeCluster);

REGISTER_OP("KungfuProposeUpdate")
    // the target global step at which change should happen,
    // must be greater than the current global step
    .Input("target_global_step: int64")
    .Input("target_version: int32")
    .Input("new_cluster_size: int32")
    // indicates if the proposal is accepted
    .Output("accepted: bool")
    // indicats if self is still in the new cluster
    .Output("keep: bool");

class KungfuProposeUpdate : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int64_t gs       = context->input(0).scalar<int64_t>()();
        const int32_t version  = context->input(1).scalar<int32_t>()();
        const int32_t new_size = context->input(2).scalar<int32_t>()();
        Tensor *accepted       = nullptr;
        Tensor *keep           = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(), &accepted));
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, MakeTensorShape(), &keep));
        _kungfu_world->ProposeUpdate(gs, std::to_string(version).c_str(),
                                     new_size, accepted->scalar<bool>().data(),
                                     keep->scalar<bool>().data());
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("KungfuProposeUpdate").Device(DEVICE_CPU),
                        KungfuProposeUpdate);

REGISTER_OP("KungfuUpdateCluster")
    .Input("version: int32")  // the cluster version
    // indicates if self is still in cluster,
    // peer should quit if exist is false
    .Output("exist: bool");

class KungfuUpdateCluster : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int32_t version = context->input(0).scalar<int32_t>()();
        Tensor *output        = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        _kungfu_world->UpdateCluster(std::to_string(version).c_str(),
                                     output->scalar<bool>().data());
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("KungfuUpdateCluster").Device(DEVICE_CPU),
                        KungfuUpdateCluster);
}  // namespace tensorflow
