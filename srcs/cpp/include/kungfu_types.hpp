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

using V = dtype;

template <> struct data_type_t<uint8_t> {
    static constexpr V value = u8;
};

template <> struct data_type_t<int32_t> {
    static constexpr V value = i32;
};

template <> struct data_type_t<int64_t> {
    static constexpr V value = i64;
};

template <> struct data_type_t<float16> {
    static constexpr V value = f16;
};

template <> struct data_type_t<float> {
    static constexpr V value = f32;
};

template <> struct data_type_t<double> {
    static constexpr V value = f64;
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

namespace internal
{
namespace ops
{
template <typename O> struct op_type_t;

using V = op;

template <> struct op_type_t<op_max> {
    static constexpr V value = max;
};

template <> struct op_type_t<op_min> {
    static constexpr V value = min;
};

template <> struct op_type_t<op_sum> {
    static constexpr V value = sum;
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
