#include <kungfu_tensorflow_ops.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>

namespace tensorflow
{
REGISTER_OP("KungfuProposeUpdate")
    // the target global step at which change should happen,
    // must be greater than the current global step
    .Input("input: int64")
    .Output("output: bool")  // indicates if the proposal is accepted
    ;

class KungfuProposeUpdate : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int64_t gs = context->input(0).scalar<int64_t>()();
        Tensor *output   = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        std::string token = std::to_string(gs);
        _kungfu_world->ProposeUpdate(token.c_str(),
                                     output->scalar<bool>().data());
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("KungfuProposeUpdate").Device(DEVICE_CPU),
                        KungfuProposeUpdate);

REGISTER_OP("KungfuUpdateCluster")
    .Input("input: int64")   // the current global step
    .Output("output: bool")  // peer should quit if output is false
    ;

class KungfuUpdateCluster : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int64_t gs = context->input(0).scalar<int64_t>()();
        Tensor *output   = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        std::string token = std::to_string(gs);
        _kungfu_world->UpdateCluster(token.c_str(),
                                     output->scalar<bool>().data());
        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("KungfuUpdateCluster").Device(DEVICE_CPU),
                        KungfuUpdateCluster);
}  // namespace tensorflow
