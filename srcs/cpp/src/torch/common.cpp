#include <kungfu/torch/common.hpp>

namespace kungfu
{
void TensorShape::AddDim(int d) { dims_.push_back(d); }

const std::vector<int64_t> &TensorShape::Dims() const { return dims_; }

std::string TensorShape::str() const
{
    std::string s = "(";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) { s += ", "; }
        s += std::to_string(dims_[i]);
    }
    s += ")";
    return s;
}

int64_t TensorShape::size() const
{
    return std::accumulate(dims_.begin(), dims_.end(), static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
}

TensorShape get_tensor_shape(torch::Tensor &x)
{
    TensorShape shape;
    for (int idx = 0; idx < x.dim(); ++idx) { shape.AddDim(x.size(idx)); }
    return shape;
}

torch::Tensor new_tensor_like(torch::Tensor input)
{
    TensorShape shape    = get_tensor_shape(input);
    auto options         = torch::TensorOptions().dtype(input.scalar_type());
    torch::Tensor output = torch::empty(shape.Dims(), options);
    return output;
}

const std::map<std::string, KungFu_Op> _kungfu_ops({
    {"sum", KungFu_SUM},
    {"min", KungFu_MIN},
    {"max", KungFu_MAX},
    {"prod", KungFu_PROD},
});

const std::map<std::string, Torch_Tensor_Type> _torch_tensor_types({
    {"torch.FloatTensor", Torch_Cpu_Float},
    {"torch.cuda.FloatTensor", Torch_Cuda_Float},
    // TODO: add more
});

void DBG(const std::string &msg)
{
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    std::cerr << msg << std::endl;
}
}  // namespace kungfu
