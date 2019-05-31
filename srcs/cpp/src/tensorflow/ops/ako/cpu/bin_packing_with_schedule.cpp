#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>

namespace tensorflow
{
REGISTER_OP("PartialNegotiatorWithSchedule")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("budget: int")
    .Attr("tensor_size: int")
    .Attr("count_gradients: int")
    .Attr("total_size: int")
    .Attr("steps: list(int)")
    .Attr("fractions: list(float)")
    .Attr("fraction: float")
    .Input("allgradients: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class PartialNegotiatorWithSchedule : public AsyncOpKernel
{
    using AsyncOpKernel::AsyncOpKernel;

    void printSchedule() {
        std::cout << "Print Schedule C++" << std::endl;
        std::cout << "Steps" << std::endl;
        for(int step : steps) {
            std::cout << "Step" << step << std::endl;
        }  
        std::cout << "Fractions" << std::endl;
        for(float f : fractions) {
            std::cout << "Fraction" << f << std::endl;
        }   
    }

  public:
    std::string input_tensor_name_;
    int32_t tensorSize_;
    int32_t count_gradients_;
    int32_t budget;
    float find_epoch_denominator_;
    float initial_fraction_;
    int32_t total_size_;

    std::vector<int> steps;
    std::vector<float> fractions;

    int repartition_id = 0;

    explicit PartialNegotiatorWithSchedule(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));

        OP_REQUIRES_OK(context, context->GetAttr("budget", &budget));
        OP_REQUIRES(context, budget > 0,
                    errors::InvalidArgument("budget must be greater than 0"));

        OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &tensorSize_));
        OP_REQUIRES(
            context, tensorSize_ > 0,
            errors::InvalidArgument("tensor size must be greater than 0"));

        OP_REQUIRES_OK(context, context->GetAttr("steps", &steps));
        OP_REQUIRES(
            context, steps.size() > 0,
            errors::InvalidArgument("steps size must be greater than 0"));

        OP_REQUIRES_OK(context, context->GetAttr("fractions", &fractions));
        OP_REQUIRES(
            context, fractions.size() > 0,
            errors::InvalidArgument("fractions size must be greater than 0"));


        OP_REQUIRES_OK(context, context->GetAttr("fraction", &initial_fraction_));
        OP_REQUIRES(
            context, initial_fraction_ > 0,
            errors::InvalidArgument("initial_fraction must be greater than 0"));

        OP_REQUIRES_OK(context,
                       context->GetAttr("count_gradients", &count_gradients_));
        OP_REQUIRES(
            context, count_gradients_ > 0,
            errors::InvalidArgument("gradient count must be greater than 0"));

        OP_REQUIRES_OK(context, context->GetAttr("total_size", &total_size_));
        OP_REQUIRES(
            context, total_size_ > 0,
            errors::InvalidArgument("total size of all gradients must be greater than 0"));

        _partial_exchange_manager->setCountGradients(count_gradients_);
        _partial_exchange_manager->setTotalSize(total_size_);
        _partial_exchange_manager->setBudget(budget);
        _partial_exchange_manager->addTensorInfo(input_tensor_name_,
                                                 tensorSize_);
        
        _partial_exchange_manager->setFraction(initial_fraction_);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        DCHECK_EQ(1, context->num_inputs());

        Tensor &gradients = (Tensor &)context->input(0);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, gradients.shape(), &output));

        int64_t gs = (int64_t) _kungfu_world->GetGlobalStep();

        if(gs == steps[repartition_id]) {
            std::cout << "Repartition ID " << repartition_id << " with fraction " << fractions[repartition_id] << std::endl;

            _partial_exchange_manager->repartition(fractions[repartition_id], repartition_id);
            repartition_id++;
        }
    

        if (_partial_exchange_manager->isReadyForNegotiation(
                input_tensor_name_, _kungfu_world->GetGlobalStep())) {
            _kungfu_world->AllReduce(gradients.tensor_data().data(),
                                     (void *)(output->tensor_data().data()),
                                     gradients.NumElements(),
                                     to_kungfu_type(gradients.dtype()),
                                     KungFu_SUM, input_tensor_name_.c_str(), done);
            // Because it is synchronous, the done callback will signal when the
            // value held
            // in the memory where output points to is ready to be used.
        } else {
            *output = gradients;
            done();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PartialNegotiatorWithSchedule").Device(DEVICE_CPU),
                        PartialNegotiatorWithSchedule);

}  // namespace tensorflow
