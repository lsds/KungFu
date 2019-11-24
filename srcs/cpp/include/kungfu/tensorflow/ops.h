// https://www.tensorflow.org/extend/adding_an_op
#pragma once
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu.h>
#include <kungfu/python/init.h>

namespace tensorflow
{
inline KungFu_Datatype to_kungfu_type(const DataType &dtype)
{
    switch (dtype) {
    case DT_INT32:
        return KungFu_INT32;
    case DT_INT64:
        return KungFu_INT64;
    case DT_BFLOAT16:
        return KungFu_FLOAT16;
    case DT_FLOAT:
        return KungFu_FLOAT;
    case DT_DOUBLE:
        return KungFu_DOUBLE;
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

template <typename... Dims> TensorShape MakeTensorShape(const Dims &... dims)
{
    std::array<int, sizeof...(Dims)> ds({static_cast<int>(dims)...});
    TensorShape shape;
    for (auto d : ds) { shape.AddDim(d); }
    return shape;
}

#define REGISTER_KUNGFU_OP(T) REGISTER_OP("Kungfu" #T)

#define REGISTER_KUNGFU_KERNEL_BUILDER(T, D)                                   \
    REGISTER_KERNEL_BUILDER(Name("Kungfu" #T).Device(D), T);
}  // namespace tensorflow
