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
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class AllReduceGpu : public AsyncOpKernel
{
    std::string input_tensor_name_;

  public:
    explicit AllReduceGpu(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(context, input_tensor_name_.size() >= 0,
                    errors::InvalidArgument(
                        "Need input_tensor_name must not be empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        kungfu::tensorflow::_world_gpu->AllReduce(
            [stream = context->op_device_context()->stream()]() {
                stream->BlockHostUntilDone();
            },
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduceGpu").Device(DEVICE_GPU), AllReduceGpu);
}  // namespace tensorflow
