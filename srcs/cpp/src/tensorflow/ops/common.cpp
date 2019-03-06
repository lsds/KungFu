#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{

REGISTER_OP("GlobalStepModifier")
    .Input("input: int32")
    .Output("output: int32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));  // TODO: don't require input
        // c->set_output(0, TensorShape());
        return Status::OK();
    });

class GlobalStepModifier : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);  // ignore input
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        int32_t *y =
            static_cast<int32_t *>((void *)output->tensor_data().data());
        y[0] = _kungfu_world->AdvanceGlobalStep();
    }
};

REGISTER_KERNEL_BUILDER(Name("GlobalStepModifier").Device(DEVICE_CPU),
                        GlobalStepModifier);

REGISTER_OP("SetNumGradients").Input("input: int32");

class SetNumGradients : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        int32_t *x = static_cast<int32_t *>((void *)input.tensor_data().data());
        _kungfu_world->SetNumGradients(x[0]);
    }
};

REGISTER_KERNEL_BUILDER(Name("SetNumGradients").Device(DEVICE_CPU),
                        SetNumGradients);

}  // namespace tensorflow
