#pragma once
#include <cstdint>

#include <kungfu/dtype.h>
#include <kungfu/op.h>
#include <kungfu/strategy.h>

namespace kungfu
{
struct float16 {
    uint16_t value;
};

namespace internal
{
namespace types
{
template <typename R> struct data_type_t;

using V = KungFu_Datatype;

template <> struct data_type_t<uint8_t> {
    static constexpr V value = KungFu_UINT8;
};

template <> struct data_type_t<int32_t> {
    static constexpr V value = KungFu_INT32;
};

template <> struct data_type_t<int64_t> {
    static constexpr V value = KungFu_INT64;
};

template <> struct data_type_t<float16> {
    static constexpr V value = KungFu_FLOAT16;
};

template <> struct data_type_t<float> {
    static constexpr V value = KungFu_FLOAT;
};

template <> struct data_type_t<double> {
    static constexpr V value = KungFu_DOUBLE;
};

struct encoding {
    using value_type = V;
    template <typename R> static constexpr value_type value()
    {
        return data_type_t<R>::value;
    }
};
}  // namespace types
}  // namespace internal

struct op_max;
struct op_min;
struct op_sum;
struct op_prod;

namespace internal
{
namespace ops
{
template <typename O> struct op_type_t;

using V = KungFu_Op;

template <> struct op_type_t<op_max> {
    static constexpr V value = KungFu_MAX;
};

template <> struct op_type_t<op_min> {
    static constexpr V value = KungFu_MIN;
};

template <> struct op_type_t<op_sum> {
    static constexpr V value = KungFu_SUM;
};

template <> struct op_type_t<op_prod> {
    static constexpr V value = KungFu_PROD;
};

struct encoding {
    using value_type = V;
    template <typename R> static constexpr value_type value()
    {
        return op_type_t<R>::value;
    }
};
}  // namespace ops
}  // namespace internal

using type_encoder = kungfu::internal::types::encoding;

using op_encoder = kungfu::internal::ops::encoding;
}  // namespace kungfu
