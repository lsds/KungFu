#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>

#include <partition.h>

namespace tensorflow
{
REGISTER_OP("PartialNegotiatorWithSchedule")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("tensor_size: int")
    .Attr("total_size: int")
    .Attr("count_gradients: int")
    .Attr("steps: list(int)")
    .Attr("fractions: list(float)")
    .Input("grads: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class PartialNegotiatorWithSchedule : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

  public:
    std::string input_tensor_name_;

    std::vector<int> steps;
    std::vector<float> fractions;

    Plan plan;

    explicit PartialNegotiatorWithSchedule(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        // Tensor info
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name", &input_tensor_name_));
        int32_t tensor_size;
        OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &tensor_size));        

        // Schedule
        OP_REQUIRES_OK(context, context->GetAttr("steps", &steps));
        OP_REQUIRES_OK(context, context->GetAttr("fractions", &fractions));

        // Global gradient tensors info
        int count_gradients;
        int total_size_bytes;
        OP_REQUIRES_OK(context, context->GetAttr("count_gradients", &count_gradients));
        OP_REQUIRES_OK(context, context->GetAttr("total_size", &total_size_bytes));

        _partial_exchange_manager->addSchedule(steps, fractions);
        _partial_exchange_manager->addGlobalTensorInfo((size_t) count_gradients, (size_t)total_size_bytes);
        _partial_exchange_manager->addTensorInfo(input_tensor_name_, tensor_size);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradients = (Tensor &)context->input(0);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        int64_t gs = (int64_t) _kungfu_world->GetGlobalStep();    

        if(gs == plan.next_repartition_step_) {
           plan = _partial_exchange_manager->repartition(gs);
        }

        //std::cout << "Current plan is: " << plan << std::endl;

        partition current_partition = plan.partitions_[gs % plan.partitions_.size()];
        if (current_partition.tensorNames.find(input_tensor_name_) != current_partition.tensorNames.end()) {
            _kungfu_world->AllReduce(gradients.tensor_data().data(),
                                     (void *)(output->tensor_data().data()),
                                     gradients.NumElements(),
                                     to_kungfu_type(gradients.dtype()),
                                     KungFu_SUM, input_tensor_name_.c_str(), done);
        } else {
            *output = gradients;
            done();
        }       
    }
};

REGISTER_KERNEL_BUILDER(Name("PartialNegotiatorWithSchedule").Device(DEVICE_CPU),
                        PartialNegotiatorWithSchedule);

}  // namespace tensorflow
