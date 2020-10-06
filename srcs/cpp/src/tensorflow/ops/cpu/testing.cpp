#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(FakeError)
    .Input("x: bool")
    .Output("y: bool")
    .SetIsStateful();

class FakeError : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const bool x = context->input(0).scalar<bool>()();
        Tensor *y    = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &y));
        y->scalar<bool>()() = x;
        if (x) {
            Status status(error::INTERNAL, "operation failed");
            OP_REQUIRES_ASYNC(context, !x, status, done);
        } else {
            done();
        }
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(FakeError, DEVICE_CPU);
}  // namespace tensorflow
