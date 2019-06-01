#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <limits>
#include <mutex>
#include <random>

class SelectionStrategy
{
  public:
    void setGlobalStep(int gs) {
        gs_ = gs;
    }
   
    virtual int getDestinationPeer() = 0;
  protected:
    int gs_;
};

class RandomSelector : public SelectionStrategy
{
    std::random_device random_device;
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist;

    int self_rank_;
    std::vector<int> ranks_;

  public:
    RandomSelector(int self_rank, std::vector<int> ranks)
        : self_rank_(self_rank), ranks_(ranks)
    {
        dist   = std::uniform_int_distribution<int>(0, ranks_.size() - 1);
    }

    int getDestinationPeer()
    {
        // Assumes self_rank already excluded from ranks_
        int destination = ranks_[dist(engine)];
        return destination;
    }
};

class RoundRobinSelector : public SelectionStrategy
{
    int self_rank_;
    std::vector<int> ranks_;

  public:
    RoundRobinSelector(int self_rank, std::vector<int> ranks)
        : self_rank_(self_rank), ranks_(ranks) {}

    int getDestinationPeer() { return ranks_[gs_ % ranks_.size()]; }
};

namespace tensorflow
{
REGISTER_OP("ModelAveraging")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("peer_selection_strategy: string")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Input("vars: NumTensors * T");

class ModelAveraging : public OpKernel
{
    using OpKernel::OpKernel;

    // My rank in the peer topology
    int self_rank_;

    // My peer ranks (excluding myself)
    std::vector<int> ranks_;

    // Peer rank selection strategy
    std::unique_ptr<SelectionStrategy> _peer_selection_strategy;

    // The vectors of the variable size (in bytes)
    std::vector<int> var_sizes_;

    // The size of each variable (in bytes)
    int var_type_size_;

    // The total size of variables (in bytes)
    int total_var_size_;

    // The pointer to the model buffer
    std::vector<unsigned char> modelBuf_;

    // The current global step
    int gs;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        if (peer_selection_strategy == "random") {
            _peer_selection_strategy.reset(new RandomSelector(self_rank_, ranks_));
        } else if (peer_selection_strategy == "roundrobin") {
            _peer_selection_strategy.reset(new RoundRobinSelector(self_rank_, ranks_));
        } else {
            throw "Unsupported peer selection strategy: " +
                peer_selection_strategy;
        }

        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ = 0;
        for (int var_size : var_sizes_) { total_var_size_ += var_size; }
    }

  public:
    explicit ModelAveraging(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0), gs(0)
    {
        check_attrs(context);
        modelBuf_ = std::vector<unsigned char>(total_var_size_ * var_type_size_);
    }

    ~ModelAveraging() {}

    void Compute(OpKernelContext *context) override
    {
        gs++;
        _peer_selection_strategy->setGlobalStep(gs);
        int destination = _peer_selection_strategy->getDestinationPeer();

        _kungfu_world->Request(destination, (void *)&modelBuf_[0], total_var_size_,
                               to_kungfu_type(context->input(0).dtype()));

        int offset = 0;
        for (int i = 0; i < var_sizes_.size(); i++) {
            Tensor &input = (Tensor &)context->input(i);
            Tensor other(input.dtype(), input.shape());
            std::copy(offset + &modelBuf_[0],
                      offset + &modelBuf_[0] + var_sizes_[i] * var_type_size_,
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
    .Attr("peer_selection_strategy: string")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Input("vars: NumTensors * T");

class AsyncModelAveraging : public OpKernel
{
    using OpKernel::OpKernel;

    int self_rank_;
    std::vector<int> ranks_;
    std::unique_ptr<SelectionStrategy> _peer_selection_strategy;

    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size_;

    std::vector<unsigned char> modelBuf_;
    std::vector<unsigned char> prefetchBuf;
    std::function<void()> prefetchCallback;

    int gs;

    std::mutex mu_;
    std::atomic<bool> isRequesting;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        if (peer_selection_strategy == "random") {
            _peer_selection_strategy.reset(new RandomSelector(self_rank_, ranks_));
        } else if (peer_selection_strategy == "roundrobin") {
            _peer_selection_strategy.reset(new RoundRobinSelector(self_rank_, ranks_));
        } else {
            throw "Unsupported peer selection strategy: " +
                peer_selection_strategy;
        }

        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ = 0;
        for (int var_size : var_sizes_) { total_var_size_ += var_size; }
    }

  public:
    explicit AsyncModelAveraging(OpKernelConstruction *context)
        : OpKernel(context), gs(0), var_type_size_(0), total_var_size_(0),
          isRequesting(false)
    {
        check_attrs(context);
        prefetchBuf = std::vector<unsigned char>(total_var_size_ * var_type_size_);
    }

    ~AsyncModelAveraging() {}

    void Compute(OpKernelContext *context) override
    {
        gs++;
        _peer_selection_strategy->setGlobalStep(gs);
        int destination = _peer_selection_strategy->getDestinationPeer();

        if (modelBuf_.empty()) {
            modelBuf_ = std::vector<unsigned char>(total_var_size_ * var_type_size_);

            _kungfu_world->Request(destination, (void *)&modelBuf_[0],
                                   total_var_size_,
                                   to_kungfu_type(context->input(0).dtype()));
            prefetchCallback = [&, modelBuf_ = modelBuf_,
                                prefetchBuf     = prefetchBuf,
                                total_var_size_ = total_var_size_,
                                var_type_size_  = var_type_size_]() {
                std::lock_guard<std::mutex> l(mu_);
                std::copy(&prefetchBuf[0],
                          &prefetchBuf[0] + total_var_size_ * var_type_size_,
                          (unsigned char *)&modelBuf_[0]);
                isRequesting = false;
            };
        }

        if (!isRequesting.load()) {
            isRequesting = true;
            _kungfu_world->Request(
                destination, (void *)&prefetchBuf[0], total_var_size_,
                to_kungfu_type(context->input(0).dtype()), prefetchCallback);
        }

        {
            std::lock_guard<std::mutex> l(mu_);
            int offset = 0;
            for (int i = 0; i < var_sizes_.size(); i++) {
                Tensor &input = (Tensor &)context->input(i);
                Tensor other(input.dtype(), input.shape());
                std::copy(offset + &modelBuf_[0],
                          offset + &modelBuf_[0] + var_sizes_[i] * var_type_size_,
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
    int total_var_size_;

    std::vector<unsigned char> modelBuf_;

    int gs;
    std::atomic<bool> isSaving;

  public:
    explicit SaveModel(OpKernelConstruction *context)
        : OpKernel(context), gs(0), var_type_size_(0), total_var_size_(0),
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

        total_var_size_ = 0;
        for (int var_size : var_sizes_) { total_var_size_ += var_size; }

        modelBuf_ = std::vector<unsigned char>(total_var_size_ * var_type_size_);
    }

    ~SaveModel() {}

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
                      &modelBuf_[0] + offset);
            offset += var_sizes_[i] * var_type_size_;
        }

        if (!isSaving.load()) {
            isSaving = true;
            std::string updateName =
                "ModelStoreUpdateAtGlobalStep " + std::to_string(gs);
            _kungfu_world->UpdateModelStore(
                updateName.c_str(), (void *)&modelBuf_[0], total_var_size_,
                to_kungfu_type(context->input(0).dtype()),
                [&]() { isSaving = false; });
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("SaveModel").Device(DEVICE_CPU), SaveModel);

REGISTER_OP("RequestModel")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("peer_selection_strategy: string")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Attr("shapes: list(shape)")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T")
    .SetShapeFn(shape_inference::UnchangedShape);

class RequestModel : public OpKernel
{
    using OpKernel::OpKernel;

    int self_rank_;
    std::vector<int> ranks_;
    std::unique_ptr<SelectionStrategy> _peer_selection_strategy;
    int num_model_vars_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size_;
    std::vector<unsigned char> modelBuf_;
    int gs;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        if (peer_selection_strategy == "random") {
            _peer_selection_strategy.reset(new RandomSelector(self_rank_, ranks_));
        } else if (peer_selection_strategy == "roundrobin") {
            _peer_selection_strategy.reset(new RoundRobinSelector(self_rank_, ranks_));
        } else {
            throw "Unsupported peer selection strategy: " +
                peer_selection_strategy;
        }

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &num_model_vars_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, shapes_.size() > 0,
                    errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ = 0;
        for (int var_size : var_sizes_) { total_var_size_ += var_size; }
    }

  public:
    explicit RequestModel(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0), gs(0)
    {
        check_attrs(context);
        modelBuf_ = std::vector<unsigned char>(total_var_size_ * var_type_size_);
    }

    ~RequestModel() {}

    void Compute(OpKernelContext *context) override
    {
        gs++;
        std::vector<Tensor *> outputs(num_model_vars_, nullptr);
        for (int i = 0; i < num_model_vars_; i++) {
            OP_REQUIRES_OK(
                context, context->allocate_output(i, shapes_[i], &outputs[i]));
        }

        _peer_selection_strategy->setGlobalStep(gs);
        int destination = _peer_selection_strategy->getDestinationPeer();

        // Fill in the model Buffer with response from random peer
        _kungfu_world->Request(destination, (void *)&modelBuf_[0], total_var_size_,
                               to_kungfu_type(context->input(0).dtype()));

        int offset = 0;
        for (int i = 0; i < var_sizes_.size(); i++) {
            std::copy(offset + &modelBuf_[0],
                      offset + &modelBuf_[0] + var_sizes_[i] * var_type_size_,
                      (unsigned char *)outputs[i]->tensor_data().data());
            offset += var_sizes_[i] * var_type_size_;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("RequestModel").Device(DEVICE_CPU), RequestModel);

REGISTER_OP("AsyncRequestModel")
    .Attr("T: {float32}")
    .Attr("self_rank: int")
    .Attr("ranks: list(int)")
    .Attr("peer_selection_strategy: string")
    .Attr("NumTensors: int")
    .Attr("var_type_size: int")
    .Attr("var_sizes: list(int)")
    .Attr("shapes: list(shape)")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T")
    .SetShapeFn(shape_inference::UnchangedShape);

class AsyncRequestModel : public OpKernel
{

    using OpKernel::OpKernel;

    int self_rank_;
    std::vector<int> ranks_;
    std::unique_ptr<SelectionStrategy> _peer_selection_strategy;
    int num_model_vars_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size_;
    std::vector<unsigned char> modelBuf_;
    std::vector<unsigned char> prefetchBuf;
    std::function<void()> prefetchCallback;
    int gs;

    std::mutex mu_;
    std::atomic<bool> isRequesting;

    void check_attrs(OpKernelConstruction *context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        if (peer_selection_strategy == "random") {
            _peer_selection_strategy.reset(new RandomSelector(self_rank_, ranks_));
        } else if (peer_selection_strategy == "roundrobin") {
            _peer_selection_strategy.reset(new RoundRobinSelector(self_rank_, ranks_));
        } else {
            throw "Unsupported peer selection strategy: " +
                peer_selection_strategy;
        }

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("NumTensors", &num_model_vars_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, shapes_.size() > 0,
                    errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ = 0;
        for (int var_size : var_sizes_) { total_var_size_ += var_size; }
    }

  public:
    explicit AsyncRequestModel(OpKernelConstruction *context)
        : OpKernel(context), gs(0), var_type_size_(0), total_var_size_(0),
          isRequesting(false)
    {
        check_attrs(context);
        prefetchBuf = std::vector<unsigned char>(total_var_size_ * var_type_size_);
    }

    ~AsyncRequestModel() {}

    void Compute(OpKernelContext *context) override
    {
        gs++;
        std::vector<Tensor *> outputs(num_model_vars_, nullptr);
        for (int i = 0; i < num_model_vars_; i++) {
            OP_REQUIRES_OK(
                context, context->allocate_output(i, shapes_[i], &outputs[i]));
        }

        _peer_selection_strategy->setGlobalStep(gs);
        int destination = _peer_selection_strategy->getDestinationPeer();

        if (modelBuf_.empty()) {
            modelBuf_ = std::vector<unsigned char>(total_var_size_ * var_type_size_);

            _kungfu_world->Request(destination, (void *)&modelBuf_[0],
                                   total_var_size_,
                                   to_kungfu_type(context->input(0).dtype()));
            prefetchCallback = [&, modelBuf_ = modelBuf_,
                                prefetchBuf     = prefetchBuf,
                                total_var_size_ = total_var_size_,
                                var_type_size_  = var_type_size_]() {
                std::lock_guard<std::mutex> l(mu_);
                std::copy(&prefetchBuf[0],
                          &prefetchBuf[0] + total_var_size_ * var_type_size_,
                          (unsigned char *)&modelBuf_[0]);
                isRequesting = false;
            };
        }

        if (!isRequesting.load()) {
            isRequesting = true;
            _kungfu_world->Request(
                destination, (void *)&prefetchBuf[0], total_var_size_,
                to_kungfu_type(context->input(0).dtype()), prefetchCallback);
        }

        {
            std::lock_guard<std::mutex> l(mu_);
            int offset = 0;
            for (int i = 0; i < var_sizes_.size(); i++) {
                std::copy(offset + &modelBuf_[0],
                          offset + &modelBuf_[0] + var_sizes_[i] * var_type_size_,
                          (unsigned char *)outputs[i]->tensor_data().data());
                offset += var_sizes_[i] * var_type_size_;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AsyncRequestModel").Device(DEVICE_CPU),
                        AsyncRequestModel);
}  // namespace tensorflow
