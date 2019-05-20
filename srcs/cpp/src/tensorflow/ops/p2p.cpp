#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("SendTo")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("rank: int32")
    .Input("input: T");

class SendTo : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &rank_tensor = context->input(0);
        int32_t rank              = rank_tensor.scalar<int32_t>()();

        const Tensor &input = context->input(1);

        LOG(INFO) << "Sending tensor " << input.DebugString()
                  << " to peer rank " << rank;
        _kungfu_world->SendTo(
            rank, input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()), name().c_str(), [] {});
    }
};

REGISTER_KERNEL_BUILDER(Name("SendTo").Device(DEVICE_CPU), SendTo);

}  // namespace tensorflow
