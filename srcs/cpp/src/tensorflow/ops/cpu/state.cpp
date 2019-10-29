#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_OP("KungfuCounter")
    .Output("count: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class KungfuCounter : public OpKernel
{
    int32_t counter_;

  public:
    explicit KungfuCounter(OpKernelConstruction *context)
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

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("KungfuCounter").Device(DEVICE_CPU),
                        KungfuCounter);
}  // namespace tensorflow
