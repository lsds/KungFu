#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
REGISTER_OP("SendTo")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("rank: int32")
    .Input("input: T");

class SendTo : public OpKernel
{
    using OpKernel::OpKernel;
    std::string input_tensor_name_;

  public:
    explicit SendTo(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &rank_tensor = context->input(0);
        int32_t rank              = rank_tensor.scalar<int32_t>()();

        const Tensor &input = context->input(1);

        _kungfu_world->SendTo(
            rank, input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()), input_tensor_name_.c_str(), [] {});
    }
};

REGISTER_KERNEL_BUILDER(Name("SendTo").Device(DEVICE_CPU), SendTo);

REGISTER_OP("MergeReceived")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class MergeReceived : public OpKernel
{
    using OpKernel::OpKernel;

    std::string input_tensor_name_;

    Tensor acc_;
    int to_merge_;
    int merged_;
    std::mutex mu_;

  public:
    explicit MergeReceived(OpKernelConstruction *context)
        : OpKernel(context), to_merge_(0), merged_(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));

        TensorShapeProto shape_;
        OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
        DataType dtype_;
        OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
        acc_ = Tensor(dtype_, shape_);

        _kungfu_world->RegisterDataCallback(
            input_tensor_name_.c_str(), [&](void *data) {
                // TODO: give priority to callback or it always lose to Compute
                std::lock_guard<std::mutex> _lk(mu_);
                to_merge_++;
                add_tensor(acc_, acc_.tensor_data().data(), data);
            });
    }

    ~MergeReceived()
    {
        std::lock_guard<std::mutex> _lk(mu_);
        _kungfu_world->UnregisterDataCallback(input_tensor_name_.c_str());
        LOG(INFO) << merged_ << " merged, " << to_merge_ << " unmerged";
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        {
            std::lock_guard<std::mutex> _lk(mu_);
            if (to_merge_ > 0) {
                merged_ += to_merge_;
                to_merge_ = 0;
                add_tensor(*output, input.tensor_data().data(),
                           acc_.tensor_data().data());
            } else {
                output->CopyFrom(input, input.shape());
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("MergeReceived").Device(DEVICE_CPU),
                        MergeReceived);

}  // namespace tensorflow
