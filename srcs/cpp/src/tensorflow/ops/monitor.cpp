#include <queue>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu/ema.hpp>
#include <kungfu_tensorflow_ops.h>

namespace tensorflow
{
// https://github.com/lsds/KungFu/blob/adaptive-batch/srcs/cpp/src/tensorflow/ops/collective.cpp#L144
REGISTER_OP("ControllerRunningSum")
    .Attr("interval: int")
    .Attr("future_batch_limit: int")
    .Input("gradient_noise: float32")
    .Output("output: int32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

class ControllerRunningSum : public OpKernel
{
    using OpKernel::OpKernel;

    int interval;
    int future_batch_limit;
    std::queue<float> noises;
    float running_sum;

  public:
    explicit ControllerRunningSum(OpKernelConstruction *context)
        : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("interval", &interval));
        OP_REQUIRES(
            context, interval >= 0,
            errors::InvalidArgument("interval must be greater than zero"));
        OP_REQUIRES_OK(context, context->GetAttr("future_batch_limit",
                                                 &future_batch_limit));
        OP_REQUIRES(context, future_batch_limit > 0,
                    errors::InvalidArgument(
                        "future batch limit must be greater than zero"));
    }

    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradient_noise_tensor = (Tensor &)context->input(0);

        float noise = (float)gradient_noise_tensor.scalar<float>()();
        noises.push(abs(noise));
        running_sum += abs(noise);

        if (noises.size() >= interval) {
            running_sum -= noises.front();
            noises.pop();
        }

        float future_batch = 0.0;
        if (noises.size() > 0) { future_batch = running_sum / noises.size(); }

        if (future_batch <= future_batch_limit) {
            LOG(INFO) << "[Running Sum] Future batch " << future_batch
                      << "; Noise " << noise;
            // noise_file << future_batch << std::endl;
        } else {
            LOG(INFO) << "Future batch limit exceeded. Capping to "
                      << future_batch_limit;
            LOG(INFO) << "[Running Sum] Future batch " << future_batch_limit
                      << "; Noise " << noise;
            // noise_file << future_batch_limit << std::endl;
            future_batch = future_batch_limit;
        }

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        output->scalar<int32_t>()() = static_cast<int32_t>(future_batch);
    }
};

REGISTER_KERNEL_BUILDER(Name("ControllerRunningSum").Device(DEVICE_CPU),
                        ControllerRunningSum);

// Unused. Experiments show that this is not effective
REGISTER_OP("ControllerEMA").Input("gradient_noise: float32");

class ControllerEMA : public OpKernel
{
    using OpKernel::OpKernel;

    int gs;
    float future_batch_ema;
    float alpha;

  public:
    void Compute(OpKernelContext *context) override
    {
        gs++;
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradient_noise_tensor = (Tensor &)context->input(0);

        float noise = abs((float)gradient_noise_tensor.scalar<float>()());

        if (future_batch_ema == 0.0) {
            future_batch_ema = noise;
        } else {
            future_batch_ema = alpha * noise + (1 - alpha) * future_batch_ema;
        }

        LOG(INFO) << "[EMA] Future batch " << future_batch_ema << "; Noise "
                  << noise;
    }
};

REGISTER_KERNEL_BUILDER(Name("ControllerEMA").Device(DEVICE_CPU),
                        ControllerEMA);

}  // namespace tensorflow
