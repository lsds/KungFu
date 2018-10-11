#include "negotiator.h"

#include <tensorflow/core/framework/op_kernel.h>

#include <cstdio>

namespace tensorflow
{

class Negotiator : public OpKernel
{
  public:
    explicit Negotiator(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        printf("%s::%s called\n", "Negotiator", __func__);

        // Grab the input tensor
        const Tensor &input = context->input(0);
        const auto &shape = input.shape();
        printf("%s :: %s\n", name().c_str(), shape.DebugString().c_str());

        // Create an output tensor
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        auto input_flat = input.flat<float>();
        auto output_flat = output->flat<float>();
        const int n = input_flat.size();

        printf("TODO: actually Negotiate the gradients with peers\n");
        for (int i = 0; i < n; i++) { output_flat(i) = input_flat(i); }
    }
};

REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_CPU), Negotiator);

}  // namespace tensorflow
