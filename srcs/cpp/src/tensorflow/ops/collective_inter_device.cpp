#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/stream_executor/stream.h>

#include <cuda_runtime.h>
#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{

REGISTER_OP("AllReduceGpuViaCpu")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AllReduceGpuViaCpu : public AsyncOpKernel
{
    std::string input_tensor_name_;

  public:
    explicit AllReduceGpuViaCpu(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);

        const int n                = input.NumElements();
        const KungFu_Datatype type = to_kungfu_type(input.dtype());
        const int data_size        = n * kungfu_type_size(type);

        char *in_cpu  = new char[data_size];
        char *out_cpu = new char[data_size];

        auto stream = context->op_device_context()->stream();
        stream->BlockHostUntilDone();

        cudaMemcpy(in_cpu, input.tensor_data().data(), data_size,
                   cudaMemcpyDeviceToHost);

        _kungfu_world->AllReduce(
            in_cpu, out_cpu, n, type, KungFu_SUM, input_tensor_name_.c_str(),
            [&] {
                cudaMemcpy(const_cast<char *>(output->tensor_data().data()),
                           out_cpu, data_size, cudaMemcpyHostToDevice);
                delete[] in_cpu;
                delete[] out_cpu;
                done();
            });
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduceGpuViaCpu").Device(DEVICE_GPU),
                        AllReduceGpuViaCpu);

}  // namespace tensorflow
