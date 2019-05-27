#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <limits>

#include <random>

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
            to_kungfu_type(input.dtype()), input_tensor_name_.c_str());
    }
};

REGISTER_KERNEL_BUILDER(Name("SendTo").Device(DEVICE_CPU), SendTo);

REGISTER_OP("RequestModel")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Attr("var_names: list(string)")
    .Attr("shapes: list(shape)")
    .Attr("dtypes: list(type)")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T")
    .SetShapeFn(shape_inference::UnchangedShape);   

class RequestModel : public OpKernel
{
    using OpKernel::OpKernel;
    std::random_device random_device;   
    std::mt19937 engine{random_device()};  

    int self_rank_;

    std::vector<int> ranks_;
    int model_size_;

    std::vector<std::string> var_names_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<DataType> dtypes_;


    int gs;

    //std::mutex mu_;

    void check_attrs(OpKernelConstruction *context) {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("var_names", &var_names_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &model_size_));

        OP_REQUIRES(
            context, var_names_.size() > 0,
            errors::InvalidArgument("var_names_ must not be empty"));
        OP_REQUIRES(
            context, shapes_.size() > 0,
            errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(
            context, dtypes_.size() > 0,
            errors::InvalidArgument("dtypes_ must not be empty"));
        OP_REQUIRES(
            context, ranks_.size() > 0,
            errors::InvalidArgument("ranks_ must not be empty"));
          
    }

  public:
    explicit RequestModel(OpKernelConstruction *context) : OpKernel(context), gs(0)
    {   
        check_attrs(context);
    }

    void Compute(OpKernelContext *context) override
    {   
        //std::cout <<  "Requesting model at global step " << gs << std::endl;
        gs++;
        //std::lock_guard<std::mutex> _lk(mu_);
        std::vector<Tensor*> outputs(model_size_, nullptr);
        for(int i = 0; i < model_size_; i++) {
            OP_REQUIRES_OK(context,
                        context->allocate_output(i, shapes_[i], &outputs[i]));
        }

        std::uniform_int_distribution<int> dist(0, ranks_.size() - 1);
        int destination = dist(engine);
        while(destination == self_rank_) { destination = dist(engine); }

        for(int i = 0; i < model_size_; i++) {
            _kungfu_world->Request(destination, 
                                         (void *) outputs[i]->tensor_data().data(),
                                         outputs[i]->NumElements(), 
                                         to_kungfu_type(outputs[i]->dtype()),
                                         var_names_[i].c_str());
        }
        //std::cout <<  "Finish requesting model at global step " << gs << std::endl;
    }

   
};

REGISTER_KERNEL_BUILDER(Name("RequestModel").Device(DEVICE_CPU), RequestModel);







REGISTER_OP("SaveModel")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("NumTensors: int")
    .Attr("var_names: list(string)")
    .Input("vars: NumTensors * T");

class SaveModel : public OpKernel
{
    using OpKernel::OpKernel;
    std::vector<std::string> var_names_;

    int gs;
  public:

    explicit SaveModel(OpKernelConstruction *context) : OpKernel(context), gs(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("var_names", &var_names_));
        OP_REQUIRES(context, var_names_.size() > 0,
                    errors::InvalidArgument("number of variable names must be greater than 0"));
    }
    void Compute(OpKernelContext *context) override
    {   
        gs++;
        OP_REQUIRES(context, context->num_inputs() > 0,
                    errors::InvalidArgument("Wrong number of inputs for operator SaveModel"));

        //std::cout << "Saving the model at global step " << gs << std::endl;

        for(int i = 0; i < var_names_.size(); i++) {
            const Tensor &input = context->input(i);
            //std::cout << "Before index " << i << input.DebugString() << std::endl;
            _kungfu_world->UpdateModelStore(var_names_[i].c_str(),
                                            input.tensor_data().data(),
                                            input.NumElements(), 
                                            to_kungfu_type(input.dtype()));
            //std::cout << "After index " << i << std::endl;
        }   
        //std::cout << "Finish saving the model at global step " << gs << std::endl;
    }

   
};

REGISTER_KERNEL_BUILDER(Name("SaveModel").Device(DEVICE_CPU), SaveModel);


}  // namespace tensorflow
