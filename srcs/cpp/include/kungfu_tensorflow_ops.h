// https://www.tensorflow.org/extend/adding_an_op
#pragma once
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu.h>
#include <kungfu_tensorflow_init.h>

namespace tensorflow
{
inline int64_t shape_size(const TensorShapeProto &shape)
{
    int64_t s      = 1;
    const int rank = shape.dim_size();
    for (int i = 0; i < rank; ++i) {
        const auto dim = shape.dim(i).size();
        if (dim < 0) { return -1; }
        s *= dim.size();
    }
    return s;
}

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
}  // namespace tensorflow
