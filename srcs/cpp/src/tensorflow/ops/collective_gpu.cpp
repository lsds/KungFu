#include <thread>

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

void spin_wait(const std::function<bool()> &cond, int ns = 100)
{
    while (!cond()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
    }
}

REGISTER_OP("AllReduceGpu")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AllReduceGpu : public AsyncOpKernel
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

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));

        auto device_context = context->op_device_context();
        auto executor       = device_context->stream()->parent();
        auto ready_event    = new perftools::gputools::Event(executor);
        ready_event->Init();
        device_context->stream()->ThenRecordEvent(ready_event);

        kungfu::tensorflow::_world_gpu->AllReduce(
            [ready_event = ready_event] {
                spin_wait([ready_event = ready_event] {
                    return ready_event->PollForStatus() !=
                           perftools::gputools::Event::Status::kPending;
                });
            },
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(),
            [=] {
                delete ready_event;
                done();
            });
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduceGpu").Device(DEVICE_GPU), AllReduceGpu);
}  // namespace tensorflow
