#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <queue>

#include <fstream>

namespace tensorflow
{
// The AllReduce operator takes a single tensor (e.g. the computed gradient),
// and reduce (by taking sum) with the peers, and finally returns a tensor with
// exactly the same shape.
REGISTER_OP("AllReduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class AllReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;

  public:
    explicit AllReduce(OpKernelConstruction *context) : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
    }

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world->AllReduce(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), KungFu_SUM,
            input_tensor_name_.c_str(), done);
    }
};

REGISTER_KERNEL_BUILDER(Name("AllReduce").Device(DEVICE_CPU), AllReduce);

REGISTER_OP("Broadcast")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class Broadcast : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        _kungfu_world->Broadcast(
            input.tensor_data().data(), (void *)(output->tensor_data().data()),
            input.NumElements(), to_kungfu_type(input.dtype()), name().c_str(),
            done);
    }
};

REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), Broadcast);

REGISTER_OP("GradientNoise")
    .Attr("alpha: float")
    .Input("g_biased: float32")
    .Input("s_biased: float32")
    .Output("output: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class GradientNoise : public OpKernel
{
    using OpKernel::OpKernel;

    float alpha_;
    float g_ema;
    float s_ema;

  public:
    explicit GradientNoise(OpKernelConstruction *context)
        : OpKernel(context), g_ema(0), s_ema(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
        OP_REQUIRES(context, alpha_ > 0,
                    errors::InvalidArgument("alpha must be greater than zero"));
    }

    void Compute(OpKernelContext *context) override
    {
        DCHECK_EQ(2, context->num_inputs());

        Tensor &g_biased_tensor = (Tensor &)context->input(0);
        Tensor &s_biased_tensor = (Tensor &)context->input(1);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, g_biased_tensor.shape(), &output));

        float g_current = (float)g_biased_tensor.scalar<float>()();
        float s_current = (float)s_biased_tensor.scalar<float>()();

        if (g_ema == 0.0) {
            g_ema = g_current;
        } else {
            g_ema = alpha_ * g_current + (1 - alpha_) * g_ema;
        }

        if (s_ema == 0.0) {
            s_ema = s_current;
        } else {
            s_ema = alpha_ * s_current + (1 - alpha_) * s_ema;
        }

        float gradient_noise = s_ema / g_ema;
        if (gradient_noise > 10000) {
            std::cout << "s_ema = " << s_ema << "; g_ema = " << g_ema << std::endl;
        }

        float *y = static_cast<float *>((void *)output->tensor_data().data());
        y[0]     = gradient_noise;
    }
};

REGISTER_KERNEL_BUILDER(Name("GradientNoise").Device(DEVICE_CPU),
                        GradientNoise);

REGISTER_OP("ControllerRunningSum")
    .Attr("worker_id: int")
    .Attr("interval: int")
    .Attr("future_batch_limit: int")
    .Input("gradient_noise: float32");

class ControllerRunningSum : public OpKernel
{
    using OpKernel::OpKernel;

    int gs;
    int interval;
    int future_batch_limit;
    std::queue<float> noises;
    float running_sum;  

    int worker_id;
    std::ofstream noise_file;

  public:
    explicit ControllerRunningSum(OpKernelConstruction *context)
        : OpKernel(context), gs(0)
    {   
        OP_REQUIRES_OK(context, context->GetAttr("worker_id", &worker_id));
        OP_REQUIRES(context, worker_id >= 0,
            errors::InvalidArgument("worker id must be non-negative"));
        
        OP_REQUIRES_OK(context, context->GetAttr("interval", &interval));
        OP_REQUIRES(context, interval >= 0,
            errors::InvalidArgument("interval must be greater than zero"));
        OP_REQUIRES_OK(context, context->GetAttr("future_batch_limit", &future_batch_limit));
        OP_REQUIRES(context, future_batch_limit > 0,
            errors::InvalidArgument("future batch limit must be greater than zero"));

        // Write noise to file
        // "/home/ab7515/noise-worker-" + std::to_string(worker_id) + ".txt";
        // std::string worker_file_name = "/home/work/user-job-dir/noise-worker-" + std::to_string(worker_id) + ".txt";
        // noise_file.open(worker_file_name);
    }

    void Compute(OpKernelContext *context) override
    {
        gs++;
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradient_noise_tensor = (Tensor &)context->input(0);

        float noise = (float)gradient_noise_tensor.scalar<float>()();
        noises.push(abs(noise));
        running_sum += abs(noise);

        if(noises.size() >= interval) {
            running_sum -= noises.front();
            noises.pop();
        }

        float future_batch = 0.0;
        if (noises.size() > 0) {
           future_batch = running_sum / noises.size();
        }

        if (future_batch <= future_batch_limit) {
            LOG(INFO) << "[Running Sum] Future batch " << future_batch << "; Noise " << noise; 
            noise_file << future_batch << std::endl;
        } else {
            LOG(INFO) << "Future batch limit exceeded. Capping to " << future_batch_limit; 
            LOG(INFO) << "[Running Sum] Future batch " << future_batch_limit << "; Noise " << noise; 
            noise_file << future_batch_limit << std::endl;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ControllerRunningSum").Device(DEVICE_CPU), ControllerRunningSum);



// Unused. Experiments show that this is not effective
REGISTER_OP("ControllerEMA")
    .Input("gradient_noise: float32");

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

        LOG(INFO) << "[EMA] Future batch " << future_batch_ema << "; Noise " << noise; 
    }
};

REGISTER_KERNEL_BUILDER(Name("ControllerEMA").Device(DEVICE_CPU), ControllerEMA);

}  // namespace tensorflow
