#pragma once
#include <experimental/type_encoder>

namespace mpi
{
namespace internal
{
namespace types
{
template <typename R> struct data_type_t;

using V = std::uint8_t;

template <> struct data_type_t<int> {
    static constexpr V value = 0;
};

template <> struct data_type_t<float> {
    static constexpr V value = 1;
};

template <> struct data_type_t<double> {
    static constexpr V value = 2;
};

struct encoding {
    using types = std::tuple<int, float, double>;

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
    using types = std::tuple<op_max, op_min, op_sum>;

    using value_type = V;
    template <typename R> static constexpr value_type value()
    {
        return op_type_t<R>::value;
    }
};
}  // namespace ops
}  // namespace internal

using type_encoder =
    std::experimental::basic_type_encoder<mpi::internal::types::encoding>;

using op_encoder =
    std::experimental::basic_type_encoder<mpi::internal::ops::encoding>;
}  // namespace mpi
