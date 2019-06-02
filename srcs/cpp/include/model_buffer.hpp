#pragma once
#include <algorithm>
#include <vector>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow
{

class ModelBuffer
{
    // the prefix sum of var_sizes_ scaled by dtype size.
    const std::vector<int> offsets_;

    std::vector<char> data_;

    static std::vector<int> _build_offsets(const std::vector<int> &var_sizes,
                                           int dtype_size)
    {
        std::vector<int> offsets(var_sizes.size() + 1);
        offsets[0] = 0;
        std::partial_sum(var_sizes.begin(), var_sizes.end(),
                         offsets.begin() + 1);
        std::transform(offsets.begin(), offsets.end(), offsets.begin(),
                       [=](int x) { return x * dtype_size; });
        return offsets;
    }

  public:
    ModelBuffer(const std::vector<int> &var_sizes, int dtype_size)
        : offsets_(_build_offsets(var_sizes, dtype_size)),
          // offsets_.size() == var_sizes_.size() + 1
          data_(offsets_.at(var_sizes.size()))
    {
    }

    void copyFrom(int i, const Tensor &t)
    {
        std::copy(t.tensor_data().begin(), t.tensor_data().end(),
                  data_.data() + offsets_[i]);
    }

    void copyTo(int i, Tensor &t) const
    {
        std::copy(data_.data() + offsets_[i], data_.data() + offsets_[i + 1],
                  (char *)(t.tensor_data().data()));
    }

    char *data() { return data_.data(); }
};

}  // namespace tensorflow
