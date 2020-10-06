#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(FakeError)
    .Input("x: int32")
    .Output("y: bool")
    .SetIsStateful();

class FakeError : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const int32_t x = context->input(0).scalar<int32_t>()();
        Tensor *y       = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &y));
        if (x != 0) {
            y->scalar<bool>()() = false;
            Status status(error::Code(x), "operation failed");
            OP_REQUIRES_ASYNC(context, false, status, done);
        } else {
            y->scalar<bool>()() = true;
            done();
        }
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(FakeError, DEVICE_CPU);
}  // namespace tensorflow
