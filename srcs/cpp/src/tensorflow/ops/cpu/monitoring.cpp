#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(EgressRates).Output("rates: float").SetIsStateful();

class EgressRates : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const int np  = kungfu::Peer::GetDefault()->Size();
        Tensor *rates = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, MakeTensorShape(np), &rates));
        kungfu::Peer::GetDefault()->GetEgressRates(rates->vec<float>().data());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(EgressRates, DEVICE_CPU);
}  // namespace tensorflow
