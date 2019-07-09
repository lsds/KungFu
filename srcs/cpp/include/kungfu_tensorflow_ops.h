// https://www.tensorflow.org/extend/adding_an_op
#pragma once
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu.h>
#include <kungfu_base.h>
#include <kungfu_tensorflow_init.h>

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

inline void add_tensor(Tensor &out, const void *a, const void *b)
{
    std_transform_2(a, b, (void *)out.tensor_data().data(), out.NumElements(),
                    to_kungfu_type(out.dtype()), KungFu_SUM);
}

template <typename... Dims> TensorShape MakeTensorShape(const Dims &... dims)
{
    TensorShape shape;
    for (auto d : {dims...}) { shape.AddDim(d); }
    return shape;
}
//
}  // namespace tensorflow
