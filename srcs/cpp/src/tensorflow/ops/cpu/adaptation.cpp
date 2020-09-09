#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(SetTree)
    .Input("tree: int32")
    .Output("success: bool")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class SetTree : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &tree = context->input(0);
        Tensor *succ       = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &succ));
        DCHECK_EQ(_default_peer->SetTree(tree.vec<int32_t>().data()), 0);
        succ->scalar<bool>()() = true;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(SetTree, DEVICE_CPU);

REGISTER_KUNGFU_OP(CalcStats)
    .SetIsStateful();

class CalcStats : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        _default_peer->CalcStats();
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(CalcStats, DEVICE_CPU);
}  // namespace tensorflow
