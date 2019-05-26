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
    .Output("outputs: NumTensors * T") // Try list(T) or list(tensor) if it does not work
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
    std::vector<Tensor> other_vars_;


    std::mutex mu_;

    void update_model_store(OpKernelContext* context) {
        OP_REQUIRES(
            context, var_names_.size() == context->num_inputs(),
            errors::InvalidArgument("Wrong number of inputs for operator"));

        std::cout << "Model store update" << std::endl;
        for(int i = 0; i < var_names_.size(); i++) {
            const Tensor &input = context->input(i);
            _kungfu_world->UpdateModelStore(i,
                                            input.tensor_data().data(),
                                            input.NumElements(), 
                                            to_kungfu_type(input.dtype()));
        }   
    }

    void check_attrs(OpKernelConstruction *context) {
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
            context, other_vars_.size() > 0,
            errors::InvalidArgument("other_vars_ must not be empty"));
          OP_REQUIRES(
            context, ranks_.size() > 0,
            errors::InvalidArgument("ranks_ must not be empty"));
    }

    void init_result_tensors(OpKernelConstruction *context) {
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &model_size_));
        
        for(int i = 0; i < model_size_; i++) {
            Tensor other_var = Tensor(dtypes_[i], shapes_[i]);
            other_var.flat<float>().setZero();
            //std::cout << "Initial other_tensors: " << other_var.DebugString()  << ", " << other_var.NumElements() << std::endl;
            other_vars_.push_back(other_var);
        }
    }

    void register_callbacks_for_variables(OpKernelConstruction *context) {
        for(int i = 0; i < var_names_.size(); i++) {
            _kungfu_world->RegisterDataCallback(
                std::to_string(i).c_str(), [&, i=i](void *data, int len) {
                    // /std::lock_guard<std::mutex> _lk(mu_);
        
                    std::cout << "Updating other_var " << i << " inside callback." << std::endl;

                    if(other_vars_[i].NumElements() != len) {
                        LOG(ERROR) << "The other tensor variable received has a different size: " << len  << " than the "
                                      "local variable: " << other_vars_[i].NumElements();
                    }

                    other_vars_[i].flat<float>().setZero();

                    std_transform_2((const float *)data, 
                                    (const void *) other_vars_[i].tensor_data().data(), 
                                    (void *) other_vars_[i].tensor_data().data(), 
                                    other_vars_[i].NumElements(), 
                                    to_kungfu_type(other_vars_[i].dtype()), 
                                    KungFu_SUM);
            });
        }
    }

  public:
    explicit RequestModel(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("var_names",
                                                 &var_names_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks",
                                                 &ranks_));
        init_result_tensors(context);
        check_attrs(context);
        _kungfu_world->InitModelStore(var_names_.size());
        register_callbacks_for_variables(context);
    }

    ~RequestModel()
    {
        //std::lock_guard<std::mutex> _lk(mu_);
        for(int i = 0; i < var_names_.size(); i++) {
            _kungfu_world->UnregisterDataCallback(std::to_string(i).c_str());
        }
    }

    void Compute(OpKernelContext *context) override
    {

        std::uniform_int_distribution<int> dist(0, ranks_.size() - 1);
        int destination = dist(engine);
        while(destination == self_rank_) { destination = dist(engine); }

        std::string req_name = "ThisIsTheUniqueModelRequestName";
        _kungfu_world->RequestModel(destination, req_name.c_str());

        //std::lock_guard<std::mutex> _lk(mu_);
        std::vector<Tensor*> outputs(other_vars_.size(), nullptr);
        for(int i = 0; i < other_vars_.size(); i++) {
            OP_REQUIRES_OK(context,
                        context->allocate_output(i, other_vars_[i].shape(), &outputs[i]));
        }

        for(int i = 0; i < other_vars_.size(); i++) {
            outputs[i]->CopyFrom(other_vars_[i], other_vars_[i].shape());   
            std::cout << "Output " << i << " is " << outputs[i]->DebugString() << std::endl;
        }

        update_model_store(context);
    
    }

   
};

REGISTER_KERNEL_BUILDER(Name("RequestModel").Device(DEVICE_CPU), RequestModel);

}  // namespace tensorflow
