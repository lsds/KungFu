#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/stream_executor/stream.h>

#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{

REGISTER_OP("StartGpuGroup").Input("input: string");

class StartGpuGroup : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        const auto t_names  = input.vec<std::string>();
        std::vector<std::string> names;
        for (int i = 0; i < t_names.size(); ++i) {
            names.push_back(t_names(i));
        }
        kungfu::tensorflow::_world_gpu->StartGroup(names);
    }
};

REGISTER_KERNEL_BUILDER(Name("StartGpuGroup").Device(DEVICE_CPU),
                        StartGpuGroup);

REGISTER_OP("AllReduceGpu")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AllReduceGpu : public OpKernel
{
    std::string input_tensor_name_;

  public:
    explicit AllReduceGpu(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        auto stream = context->op_device_context()->stream();
        stream->BlockHostUntilDone();

        kungfu::tensorflow::_world_gpu->AllReduce(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str());
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduceGpu").Device(DEVICE_GPU), AllReduceGpu);
}  // namespace tensorflow
