#include <cstdio>

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>

#include <algo.h>
#include <kungfu_types.hpp>

template <typename Args, typename Op>
void _call_std_transform(const Args &args, const Op &op)
{
    std::transform(std::get<0>(args), std::get<1>(args), std::get<2>(args),
                   std::get<3>(args), op);
}

template <typename T>
void std_transform_2_tpl(const void *input_1, const void *input_2, void *output,
                         int n, int binary_op)
{
    const auto args = std::make_tuple(reinterpret_cast<const T *>(input_1),
                                      reinterpret_cast<const T *>(input_1) + n,
                                      reinterpret_cast<const T *>(input_2),
                                      reinterpret_cast<T *>(output));
    switch (binary_op) {
    case kungfu::op_encoder::value<kungfu::op_sum>():
        _call_std_transform(args, std::plus<T>());
        return;
    case kungfu::op_encoder::value<kungfu::op_min>():
        _call_std_transform(args, [](T x, T y) { return std::min<T>(x, y); });
        return;
    case kungfu::op_encoder::value<kungfu::op_max>():
        _call_std_transform(args, [](T x, T y) { return std::max<T>(x, y); });
        return;
    default:
        throw std::invalid_argument(
            "std_transform_2 doesn't support binary_op: " +
            std::to_string(binary_op));
    }
}

void std_transform_2(const void *input_1, const void *input_2, void *output,
                     int n, int dtype, int binary_op)
{
    switch (dtype) {
    case kungfu::type_encoder::value<int>():
        std_transform_2_tpl<int>(input_1, input_2, output, n, binary_op);
        return;
    case kungfu::type_encoder::value<float>():
        std_transform_2_tpl<float>(input_1, input_2, output, n, binary_op);
        return;
    default:
        throw std::invalid_argument("std_transform_2 doesn't support dtype: " +
                                    std::to_string(dtype));
    }
}
