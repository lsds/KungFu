#include "negotiator.h"

#include <tensorflow/core/framework/op_kernel.h>

#include "communicator.h"

namespace tensorflow
{

class Negotiator : public AsyncOpKernel
{
  public:
    explicit Negotiator(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        const auto &shape = input.shape();
        LOG(INFO) << name() << " :: " << shape.DebugString();

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        // FIXME: support other data types
        const auto input_flat = input.flat<float>();
        auto output_flat = output->flat<float>();

        const int n = input_flat.size();
        auto a = Agent::get_instance();
        a->push(name(), input_flat.data(), n * sizeof(float));
        a->pull(name(), output_flat.data(), n * sizeof(float));

        done();  // TODO: call it async
    }
};

REGISTER_KERNEL_BUILDER(Name("Negotiator").Device(DEVICE_CPU), Negotiator);

}  // namespace tensorflow
