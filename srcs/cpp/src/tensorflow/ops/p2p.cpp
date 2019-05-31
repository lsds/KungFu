#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <limits>
#include <mutex>
#include <random>

namespace tensorflow
{
REGISTER_OP("ModelAveraging")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Input("vars: NumTensors * T");

class ModelAveraging : public OpKernel
{
    using OpKernel::OpKernel;

    std::random_device random_device;
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist;

    int self_rank_;
    std::vector<int> ranks_;

    // Used for the buffer
    std::vector<int>
        var_sizes_;  // The vector of the numbers of weights in each variable
    int var_type_size_;
    int total_var_size;  // The total number of elements of all variables.

    unsigned char *modelBuf;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size = 0;
        for (int s : var_sizes_) { total_var_size += s; }
    }

  public:
    explicit ModelAveraging(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size(0),
          modelBuf(nullptr)
    {
        check_attrs(context);
        modelBuf = (unsigned char *)malloc(total_var_size * var_type_size_);
        dist     = std::uniform_int_distribution<int>(0, ranks_.size() - 1);
    }

    ~ModelAveraging() { free(modelBuf); }

    void Compute(OpKernelContext *context) override
    {
        int destination = dist(engine);
        while (destination == self_rank_) { destination = dist(engine); }

        // Fill in the model Buffer with response from random peer
        _kungfu_world->Request(destination, (void *)modelBuf, total_var_size,
                               to_kungfu_type(context->input(0).dtype()));

        int offset = 0;
        for (int i = 0; i < var_sizes_.size(); i++) {
            Tensor &input = (Tensor &)context->input(i);
            Tensor other(input.dtype(), input.shape());
            std::copy(offset + modelBuf,
                      offset + modelBuf + var_sizes_[i] * var_type_size_,
                      (unsigned char *)other.tensor_data().data());
            auto other_flt = other.flat<float>();
            other_flt      = 0.5 * (input.flat<float>() + other.flat<float>());
            std::copy((unsigned char *)other.tensor_data().data(),
                      (unsigned char *)other.tensor_data().data() +
                          var_sizes_[i] * var_type_size_,
                      (unsigned char *)input.tensor_data().data());
            offset += var_sizes_[i] * var_type_size_;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ModelAveraging").Device(DEVICE_CPU),
                        ModelAveraging);

REGISTER_OP("AsyncModelAveraging")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Input("vars: NumTensors * T");

class AsyncModelAveraging : public OpKernel
{

    using OpKernel::OpKernel;

    std::random_device random_device;
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist;

    int self_rank_;

    std::vector<int> ranks_;

    // Used for the buffer
    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size;

    unsigned char *modelBuf;
    unsigned char *prefetchBuf;
    std::function<void()> prefetchCallback;

    int gs;

    std::mutex mu_;
    std::atomic<bool> isRequesting;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size = 0;
        for (int s : var_sizes_) { total_var_size += s; }
    }

  public:
    explicit AsyncModelAveraging(OpKernelConstruction *context)
        : OpKernel(context), gs(0), var_type_size_(0), total_var_size(0),
          modelBuf(nullptr), isRequesting(false)
    {
        check_attrs(context);
        prefetchBuf = (unsigned char *)malloc(total_var_size * var_type_size_);
        dist        = std::uniform_int_distribution<int>(0, ranks_.size() - 1);
    }

    ~AsyncModelAveraging() { free(modelBuf); }

    void Compute(OpKernelContext *context) override
    {
        gs++;

        int destination = dist(engine);
        while (destination == self_rank_) { destination = dist(engine); }

        // Fill in the model Buffer with response from random peer
        if (modelBuf == nullptr) {
            modelBuf = (unsigned char *)malloc(total_var_size * var_type_size_);

            _kungfu_world->Request(destination, (void *)modelBuf,
                                   total_var_size,
                                   to_kungfu_type(context->input(0).dtype()));
            prefetchCallback = [&, modelBuf = modelBuf,
                                prefetchBuf    = prefetchBuf,
                                total_var_size = total_var_size,
                                var_type_size_ = var_type_size_]() {
                std::lock_guard<std::mutex> l(mu_);
                std::copy(prefetchBuf,
                          prefetchBuf + total_var_size * var_type_size_,
                          (unsigned char *)modelBuf);
                isRequesting = false;
            };
        }

        if (!isRequesting.load()) {
            // no other goroutine spawned in background
            isRequesting = true;
            _kungfu_world->Request(
                destination, (void *)prefetchBuf, total_var_size,
                to_kungfu_type(context->input(0).dtype()), prefetchCallback);
        }

        {
            std::lock_guard<std::mutex> l(mu_);
            int offset = 0;
            for (int i = 0; i < var_sizes_.size(); i++) {
                // This input tensor does not seem to be initialized?
                Tensor &input = (Tensor &)context->input(i);
                Tensor other(input.dtype(), input.shape());
                std::copy(offset + modelBuf,
                          offset + modelBuf + var_sizes_[i] * var_type_size_,
                          (unsigned char *)other.tensor_data().data());
                auto other_flt = other.flat<float>();
                other_flt = 0.5 * (input.flat<float>() + other.flat<float>());
                std::copy((unsigned char *)other.tensor_data().data(),
                          (unsigned char *)other.tensor_data().data() +
                              var_sizes_[i] * var_type_size_,
                          (unsigned char *)input.tensor_data().data());
                offset += var_sizes_[i] * var_type_size_;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AsyncModelAveraging").Device(DEVICE_CPU),
                        AsyncModelAveraging);

REGISTER_OP("SaveModel")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Input("vars: NumTensors * T");

class SaveModel : public OpKernel
{
    using OpKernel::OpKernel;

    std::vector<int> var_sizes_;

    int var_type_size_;
    int total_var_size;

    unsigned char *modelBuf;

    int gs;
    std::atomic<bool> isSaving;

  public:
    explicit SaveModel(OpKernelConstruction *context)
        : OpKernel(context), gs(0), var_type_size_(0), total_var_size(0),
          isSaving(false)
    {
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES(context, var_sizes_.size() > 0,
                    errors::InvalidArgument(
                        "number of variable sizes must be greater than 0"));

        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));
        OP_REQUIRES(context, var_type_size_ > 0,
                    errors::InvalidArgument(
                        "data type size in bytes must be greater than 0"));

        // number of floats it has
        total_var_size = 0;
        for (int s : var_sizes_) { total_var_size += s; }

        modelBuf = (unsigned char *)malloc(total_var_size * var_type_size_);
    }

    ~SaveModel() { free(modelBuf); }

    void Compute(OpKernelContext *context) override
    {
        gs++;
        OP_REQUIRES(context, context->num_inputs() > 0,
                    errors::InvalidArgument(
                        "Wrong number of inputs for operator SaveModel"));

        int offset = 0;
        for (int i = 0; i < var_sizes_.size(); i++) {
            const Tensor &input = context->input(i);
            std::copy((unsigned char *)input.tensor_data().data(),
                      (unsigned char *)input.tensor_data().data() +
                          var_sizes_[i] * var_type_size_,
                      modelBuf + offset);
            offset += var_sizes_[i] * var_type_size_;
        }

        if (!isSaving.load()) {
            isSaving = true;
            std::string updateName =
                "ModelStoreUpdateAtGlobalStep " + std::to_string(gs);
            _kungfu_world->UpdateModelStore(
                updateName.c_str(), (void *)modelBuf,
                total_var_size,  // how many elements of the type below it has?
                to_kungfu_type(context->input(0).dtype()),
                [&]() { isSaving = false; });  // do not put nullptr!
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("SaveModel").Device(DEVICE_CPU), SaveModel);

REGISTER_OP("RequestModel")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Attr("type_size_bytes: int")
    .Attr("var_sizes: list(int)")
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
    std::uniform_int_distribution<int> dist;

    int self_rank_;

    std::vector<int> ranks_;
    int model_size_;

    std::vector<std::string> var_names_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<DataType> dtypes_;

    // Used for the buffer
    std::vector<int> var_sizes_;
    int type_size_bytes_;
    int total_buf_size_;

    unsigned char *modelBuf;

    // std::mutex mu_;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("var_names", &var_names_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &model_size_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("type_size_bytes", &type_size_bytes_));

        OP_REQUIRES(context, var_names_.size() > 0,
                    errors::InvalidArgument("var_names_ must not be empty"));
        OP_REQUIRES(context, shapes_.size() > 0,
                    errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(context, dtypes_.size() > 0,
                    errors::InvalidArgument("dtypes_ must not be empty"));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_buf_size_ = 0;
        for (int s : var_sizes_) { total_buf_size_ += s; }
    }

  public:
    explicit RequestModel(OpKernelConstruction *context)
        : OpKernel(context), type_size_bytes_(0), total_buf_size_(0),
          modelBuf(nullptr)
    {
        check_attrs(context);
        modelBuf = (unsigned char *)malloc(total_buf_size_ * type_size_bytes_);
        dist     = std::uniform_int_distribution<int>(0, ranks_.size() - 1);
    }

    ~RequestModel() { free(modelBuf); }

    void Compute(OpKernelContext *context) override
    {
        std::vector<Tensor *> outputs(model_size_, nullptr);
        for (int i = 0; i < model_size_; i++) {
            OP_REQUIRES_OK(
                context, context->allocate_output(i, shapes_[i], &outputs[i]));
        }

        int destination = dist(engine);
        while (destination == self_rank_) { destination = dist(engine); }

        // Fill in the model Buffer with response from random peer
        _kungfu_world->Request(destination, (void *)modelBuf, total_buf_size_,
                               to_kungfu_type(context->input(0).dtype()));

        int offset = 0;
        for (int i = 0; i < var_sizes_.size(); i++) {
            std::copy(offset + modelBuf,
                      offset + modelBuf + var_sizes_[i] * type_sizßßße_bytes_,
                      (unsigned char *)outputs[i]->tensor_data().data());
            offset += var_sizes_[i] * type_size_bytes_;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("RequestModel").Device(DEVICE_CPU), RequestModel);

REGISTER_OP("AsyncRequestModel")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("NumTensors: int")
    .Attr("type_size_bytes: int")
    .Attr("var_sizes: list(int)")
    .Attr("var_names: list(string)")
    .Attr("shapes: list(shape)")
    .Attr("dtypes: list(type)")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AsyncRequestModel : public OpKernel
{

    using OpKernel::OpKernel;

    std::random_device random_device;
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist;

    int self_rank_;

    std::vector<int> ranks_;
    int model_size_;

    std::vector<std::string> var_names_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<DataType> dtypes_;

    // Used for the buffer
    std::vector<int> var_sizes_;
    int type_size_bytes_;
    int total_buf_size_;

    unsigned char *modelBuf;
    unsigned char *prefetchBuf;
    std::function<void()> prefetchCallback;

    int gs;

    std::mutex mu_;
    std::atomic<bool> alreadyRequesting;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("var_names", &var_names_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &model_size_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("type_size_bytes", &type_size_bytes_));

        OP_REQUIRES(context, var_names_.size() > 0,
                    errors::InvalidArgument("var_names_ must not be empty"));
        OP_REQUIRES(context, shapes_.size() > 0,
                    errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(context, dtypes_.size() > 0,
                    errors::InvalidArgument("dtypes_ must not be empty"));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_buf_size_ = 0;
        for (int s : var_sizes_) { total_buf_size_ += s; }
    }

  public:
    explicit AsyncRequestModel(OpKernelConstruction *context)
        : OpKernel(context), gs(0), type_size_bytes_(0), total_buf_size_(0),
          modelBuf(nullptr), alreadyRequesting(false)
    {
        check_attrs(context);
        prefetchBuf =
            (unsigned char *)malloc(total_buf_size_ * type_size_bytes_);
        dist = std::uniform_int_distribution<int>(0, ranks_.size() - 1);
    }

    ~AsyncRequestModel() { free(modelBuf); }

    void Compute(OpKernelContext *context) override
    {
        gs++;
        std::vector<Tensor *> outputs(model_size_, nullptr);
        for (int i = 0; i < model_size_; i++) {
            OP_REQUIRES_OK(
                context, context->allocate_output(i, shapes_[i], &outputs[i]));
        }
        // int destination = dist(engine);
        // while (destination == self_rank_) { destination = dist(engine); }

        // ranks_ do not include self rank
        int destination = ranks_[gs % ranks_.size()];

        // Fill in the model Buffer with response from random peer
        if (modelBuf == nullptr) {
            modelBuf =
                (unsigned char *)malloc(total_buf_size_ * type_size_bytes_);

            _kungfu_world->Request(destination, (void *)modelBuf,
                                   total_buf_size_,
                                   to_kungfu_type(context->input(0).dtype()));
            prefetchCallback = [&, modelBuf = modelBuf,
                                prefetchBuf      = prefetchBuf,
                                total_buf_size_  = total_buf_size_,
                                type_size_bytes_ = type_size_bytes_]() {
                std::lock_guard<std::mutex> l(mu_);
                std::copy(prefetchBuf,
                          prefetchBuf + total_buf_size_ * type_size_bytes_,
                          (unsigned char *)modelBuf);
                alreadyRequesting = false;
            };
        }

        if (!alreadyRequesting.load()) {
            // no other goroutine spawned in background
            alreadyRequesting = true;
            _kungfu_world->Request(
                destination, (void *)prefetchBuf, total_buf_size_,
                to_kungfu_type(context->input(0).dtype()), prefetchCallback);
        }

        {
            std::lock_guard<std::mutex> l(mu_);
            int offset = 0;
            for (int i = 0; i < var_sizes_.size(); i++) {
                std::copy(offset + modelBuf,
                          offset + modelBuf + var_sizes_[i] * type_size_bytes_,
                          (unsigned char *)outputs[i]->tensor_data().data());
                offset += var_sizes_[i] * type_size_bytes_;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AsyncRequestModel").Device(DEVICE_CPU),
                        AsyncRequestModel);
}  // namespace tensorflow
