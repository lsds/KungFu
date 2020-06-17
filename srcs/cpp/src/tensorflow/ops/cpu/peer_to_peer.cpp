#include <limits>
#include <mutex>
#include <random>

#include <kungfu/tensorflow/model_buffer.hpp>
#include <kungfu/tensorflow/ops.h>

class SelectionStrategy
{
  public:
    virtual ~SelectionStrategy() {}

    virtual int Next() = 0;

    // Factory method
    static SelectionStrategy *Create(const std::string &name,
                                     const std::vector<int> &values);
};

class RandomSelector : public SelectionStrategy
{
    std::random_device random_device_;
    std::mt19937 engine_{random_device_()};
    std::uniform_int_distribution<int> dist_;

    const std::vector<int> values_;

  public:
    RandomSelector(const std::vector<int> &values)
        : dist_(std::uniform_int_distribution<int>(0, values.size() - 1)),
          values_(values)
    {
    }

    int Next() { return values_.at(dist_(engine_)); }
};

class RoundRobinSelector : public SelectionStrategy
{
    int t_;
    const std::vector<int> values_;

  public:
    RoundRobinSelector(const std::vector<int> &values) : t_(0), values_(values)
    {
    }

    int Next()
    {
        const int now = t_;
        t_            = (t_ + 1) % values_.size();
        return values_.at(now);
    }
};

SelectionStrategy *SelectionStrategy::Create(const std::string &name,
                                             const std::vector<int> &values)
{
    if (name == "random") {
        return new RandomSelector(values);
    } else if (name == "roundrobin") {
        return new RoundRobinSelector(values);
    } else {
        throw std::invalid_argument("unsupported selection strategy: " + name);
    }
}

namespace tensorflow
{
#define MODEL_NAME "all-grads"  // TODO: make it an Attr

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
    std::unique_ptr<SelectionStrategy> peer_selection_strategy_;

    // The vectors of the variable size (in bytes)
    std::vector<int> var_sizes_;

    // The size of each variable (in bytes)
    int var_type_size_;

    // The total size of variables (in bytes)
    int total_var_size_;

    // The pointer to the model buffer
    std::unique_ptr<ModelBuffer> model_buf_;

  public:
    explicit ModelAveraging(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks must not be empty"));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        peer_selection_strategy_.reset(
            SelectionStrategy::Create(peer_selection_strategy, ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ =
            std::accumulate(var_sizes_.begin(), var_sizes_.end(), 0);
        model_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));
    }

    void Compute(OpKernelContext *context) override
    {
        int destination = peer_selection_strategy_->Next();

        _default_peer->Request(destination, MODEL_NAME, model_buf_->data(),
                               total_var_size_,
                               to_kungfu_type(context->input(0).dtype()));

        for (size_t i = 0; i < var_sizes_.size(); i++) {
            const Tensor &input = context->input(i);
            Tensor other(input.dtype(), input.shape());
            model_buf_->copyTo(i, other);
            auto other_flt = other.flat<float>();
            other_flt      = 0.5 * (input.flat<float>() + other.flat<float>());
            std::copy(other.tensor_data().begin(), other.tensor_data().end(),
                      const_cast<char *>(input.tensor_data().begin()));
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
    std::unique_ptr<SelectionStrategy> peer_selection_strategy_;

    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size_;

    std::unique_ptr<ModelBuffer> model_buf_;
    std::unique_ptr<ModelBuffer> prefetch_buf_;
    std::function<void()> prefetch_callback_;

    std::mutex mu_;
    std::atomic<bool> is_requesting_;

  public:
    explicit AsyncModelAveraging(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0),
          is_requesting_(false)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks must not be empty"));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        peer_selection_strategy_.reset(
            SelectionStrategy::Create(peer_selection_strategy, ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ =
            std::accumulate(var_sizes_.begin(), var_sizes_.end(), 0);
        prefetch_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));
    }

    void Compute(OpKernelContext *context) override
    {
        int destination = peer_selection_strategy_->Next();

        if (model_buf_.get() == nullptr) {
            model_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));

            _default_peer->Request(destination, MODEL_NAME, model_buf_->data(),
                                   total_var_size_,
                                   to_kungfu_type(context->input(0).dtype()));
            prefetch_callback_ = [&, mb = model_buf_.get(),
                                  pb              = prefetch_buf_.get(),
                                  total_var_size_ = total_var_size_,
                                  var_type_size_  = var_type_size_]() {
                std::lock_guard<std::mutex> l(mu_);
                std::copy(pb->data(),
                          pb->data() + total_var_size_ * var_type_size_,
                          mb->data());
                is_requesting_ = false;
            };
        }

        if (!is_requesting_.load()) {
            is_requesting_ = true;
            _default_peer->Request(
                destination, MODEL_NAME, prefetch_buf_->data(), total_var_size_,
                to_kungfu_type(context->input(0).dtype()), prefetch_callback_);
        }

        {
            std::lock_guard<std::mutex> l(mu_);
            for (size_t i = 0; i < var_sizes_.size(); i++) {
                const Tensor &input = context->input(i);
                Tensor other(input.dtype(), input.shape());
                model_buf_->copyTo(i, other);
                auto other_flt = other.flat<float>();
                other_flt = 0.5 * (input.flat<float>() + other.flat<float>());
                std::copy(other.tensor_data().begin(),
                          other.tensor_data().end(),
                          const_cast<char *>(input.tensor_data().begin()));
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

    std::unique_ptr<ModelBuffer> model_buf_;

    std::atomic<bool> is_saving_;

    int gs_;

  public:
    explicit SaveModel(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0),
          is_saving_(false), gs_(0)
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

        total_var_size_ =
            std::accumulate(var_sizes_.begin(), var_sizes_.end(), 0);
        model_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));
    }

    void Compute(OpKernelContext *context) override
    {
        OP_REQUIRES(context, context->num_inputs() > 0,
                    errors::InvalidArgument(
                        "Wrong number of inputs for operator SaveModel"));
        ++gs_;

        for (size_t i = 0; i < var_sizes_.size(); i++) {
            const Tensor &input = context->input(i);
            model_buf_->copyFrom(i, input);
        }

        if (!is_saving_.load()) {
            is_saving_ = true;
            std::string updateName =
                "ModelStoreUpdateAtGlobalStep " + std::to_string(gs_);
            _default_peer->Save(MODEL_NAME, model_buf_->data(), total_var_size_,
                                to_kungfu_type(context->input(0).dtype()),
                                [&] { is_saving_ = false; });
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
    std::unique_ptr<SelectionStrategy> peer_selection_strategy_;
    int num_model_vars_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size_;
    std::unique_ptr<ModelBuffer> model_buf_;

  public:
    explicit RequestModel(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks must not be empty"));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        peer_selection_strategy_.reset(
            SelectionStrategy::Create(peer_selection_strategy, ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("NumTensors", &num_model_vars_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, shapes_.size() > 0,
                    errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ =
            std::accumulate(var_sizes_.begin(), var_sizes_.end(), 0);
        model_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));
    }

    void Compute(OpKernelContext *context) override
    {
        std::vector<Tensor *> outputs(num_model_vars_, nullptr);
        for (int i = 0; i < num_model_vars_; i++) {
            OP_REQUIRES_OK(
                context, context->allocate_output(i, shapes_[i], &outputs[i]));
        }

        int destination = peer_selection_strategy_->Next();

        // Fill in the model Buffer with response from random peer
        _default_peer->Request(destination, MODEL_NAME,
                               (void *)model_buf_->data(), total_var_size_,
                               to_kungfu_type(context->input(0).dtype()));

        for (size_t i = 0; i < var_sizes_.size(); i++) {
            model_buf_->copyTo(i, *outputs[i]);
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
    std::unique_ptr<SelectionStrategy> peer_selection_strategy_;
    int num_model_vars_;
    std::vector<TensorShapeProto> shapes_;
    std::vector<int> var_sizes_;
    int var_type_size_;
    int total_var_size_;
    std::unique_ptr<ModelBuffer> model_buf_;
    std::unique_ptr<ModelBuffer> prefetch_buf_;
    std::function<void()> prefetch_callback_;

    std::mutex mu_;
    std::atomic<bool> is_requesting_;

  public:
    explicit AsyncRequestModel(OpKernelConstruction *context)
        : OpKernel(context), var_type_size_(0), total_var_size_(0),
          is_requesting_(false)
    {
        OP_REQUIRES_OK(context, context->GetAttr("self_rank", &self_rank_));
        OP_REQUIRES_OK(context, context->GetAttr("ranks", &ranks_));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks must not be empty"));
        std::string peer_selection_strategy;
        OP_REQUIRES_OK(context, context->GetAttr("peer_selection_strategy",
                                                 &peer_selection_strategy));
        peer_selection_strategy_.reset(
            SelectionStrategy::Create(peer_selection_strategy, ranks_));

        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("NumTensors", &num_model_vars_));

        // Used for the buffer
        OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("var_type_size", &var_type_size_));

        OP_REQUIRES(context, shapes_.size() > 0,
                    errors::InvalidArgument("shapes_ must not be empty"));
        OP_REQUIRES(context, ranks_.size() > 0,
                    errors::InvalidArgument("ranks_ must not be empty"));

        total_var_size_ =
            std::accumulate(var_sizes_.begin(), var_sizes_.end(), 0);
        prefetch_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));
    }

    void Compute(OpKernelContext *context) override
    {
        std::vector<Tensor *> outputs(num_model_vars_, nullptr);
        for (int i = 0; i < num_model_vars_; i++) {
            OP_REQUIRES_OK(
                context, context->allocate_output(i, shapes_[i], &outputs[i]));
        }

        int destination = peer_selection_strategy_->Next();

        if (model_buf_.get() == nullptr) {
            model_buf_.reset(new ModelBuffer(var_sizes_, var_type_size_));

            _default_peer->Request(destination, MODEL_NAME, model_buf_->data(),
                                   total_var_size_,
                                   to_kungfu_type(context->input(0).dtype()));
            prefetch_callback_ = [&, mb = model_buf_.get(),
                                  pb              = prefetch_buf_.get(),
                                  total_var_size_ = total_var_size_,
                                  var_type_size_  = var_type_size_]() {
                std::lock_guard<std::mutex> l(mu_);
                std::copy(pb->data(),
                          pb->data() + total_var_size_ * var_type_size_,
                          mb->data());
                is_requesting_ = false;
            };
        }

        if (!is_requesting_.load()) {
            is_requesting_ = true;
            _default_peer->Request(
                destination, MODEL_NAME, prefetch_buf_->data(), total_var_size_,
                to_kungfu_type(context->input(0).dtype()), prefetch_callback_);
        }

        {
            std::lock_guard<std::mutex> l(mu_);
            for (size_t i = 0; i < var_sizes_.size(); i++) {
                model_buf_->copyTo(i, *outputs[i]);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AsyncRequestModel").Device(DEVICE_CPU),
                        AsyncRequestModel);
}  // namespace tensorflow
