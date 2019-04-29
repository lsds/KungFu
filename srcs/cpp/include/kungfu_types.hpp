#pragma once
#include <cstdint>

namespace kungfu
{
struct float16 {
    std::uint16_t value
};

namespace internal
{
namespace types
{
template <typename R> struct data_type_t;

using V = std::uint8_t;

template <> struct data_type_t<uint8_t> {
    static constexpr V value = 1;
};

template <> struct data_type_t<int32_t> {
    static constexpr V value = 2;
};

template <> struct data_type_t<int64_t> {
    static constexpr V value = 3;
};

template <> struct data_type_t<float16> {
    static constexpr V value = 4;
};

template <> struct data_type_t<float> {
    static constexpr V value = 5;
};

template <> struct data_type_t<double> {
    static constexpr V value = 6;
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

using V = std::uint8_t;

template <> struct op_type_t<op_max> {
    static constexpr V value = 1;
};

template <> struct op_type_t<op_min> {
    static constexpr V value = 2;
};

template <> struct op_type_t<op_sum> {
    static constexpr V value = 3;
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
