#include <iostream>
#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
REGISTER_KUNGFU_OP(FpgaAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class FpgaAllReduce : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        // TODO: Replace the following all-reduce with your FPGA all-reduce c++
        // function. You don't need to change GO code.
        std::cerr << "TODO: call FPGA all reduce c++ functions." << std::endl;
        _kungfu_world->AllReduce(
            input.tensor_data().data(),
            const_cast<char *>(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            name().c_str());
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(FpgaAllReduce, DEVICE_CPU);
}  // namespace tensorflow
