#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(Counter)
    .Output("count: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class Counter : public OpKernel
{
    int32_t counter_;

  public:
    explicit Counter(OpKernelConstruction *context)
        : OpKernel(context), counter_(0)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        Tensor *count = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &count));
        count->scalar<int32_t>()() = counter_++;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(Counter, DEVICE_CPU);
}  // namespace tensorflow
