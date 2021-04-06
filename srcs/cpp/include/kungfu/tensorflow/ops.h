// https://www.tensorflow.org/extend/adding_an_op
#pragma once
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu.h>
#include <kungfu/nccl/common.hpp>
#include <kungfu/python/c_api.h>

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
    case DT_BOOL:
        return KungFu_BOOL;
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

template <typename... Dims>
TensorShape MakeTensorShape(const Dims &... dims)
{
    const std::array<int, sizeof...(Dims)> ds({static_cast<int>(dims)...});
    TensorShape shape;
    for (auto d : ds) { shape.AddDim(d); }
    return shape;
}

inline TensorShape BatchTensorShape(const TensorShape &shape, const int bs)
{
    TensorShape new_shape(shape);
    new_shape.InsertDim(0, bs);
    return new_shape;
}

#define REGISTER_KUNGFU_OP(T)                                                  \
    REGISTER_OP("Kungfu" #T).Attr("debug: bool = false")

#define REGISTER_KUNGFU_KERNEL_BUILDER(T, D)                                   \
    REGISTER_KERNEL_BUILDER(Name("Kungfu" #T).Device(D), T);

inline kungfu::Workspace make_workspace(const Tensor &input, Tensor *output)
{
    return kungfu::Workspace{
        .sendbuf = input.tensor_data().data(),
        .recvbuf = const_cast<char *>(output->tensor_data().data()),
        .count   = int(input.NumElements()),
        .dtype   = to_kungfu_type(input.dtype()),
    };
}
}  // namespace tensorflow
