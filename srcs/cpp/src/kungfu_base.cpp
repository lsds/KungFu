#include <cstdio>
#include <stdexcept>

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>

#include <kungfu_base.h>
#include <kungfu_types.hpp>

void invoke_callback(callback_t *f) { (*f)(); }

void delete_callback(callback_t *f) { delete f; }

// TODO: use std::apply from c++17

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

template <typename T, typename Args>
void _call_std_transform_2_tpl(const Args &args)
{
    std_transform_2_tpl<T>(std::get<0>(args), std::get<1>(args),
                           std::get<2>(args), std::get<3>(args),
                           std::get<4>(args));
}

void std_transform_2(const void *input_1, const void *input_2, void *output,
                     int n, int dtype, int binary_op)
{
    if (dtype == kungfu::type_encoder::value<kungfu::float16>() &&
        binary_op == kungfu::op_encoder::value<kungfu::op_sum>()) {
        float16_sum(output, input_1, input_2, n);
        return;
    }

    const auto args = std::make_tuple(input_1, input_2, output, n, binary_op);
    switch (dtype) {
    case kungfu::type_encoder::value<uint8_t>():
        _call_std_transform_2_tpl<uint8_t>(args);
        return;
    case kungfu::type_encoder::value<int32_t>():
        _call_std_transform_2_tpl<int32_t>(args);
        return;
    case kungfu::type_encoder::value<int64_t>():
        _call_std_transform_2_tpl<int64_t>(args);
        return;
    case kungfu::type_encoder::value<float>():
        _call_std_transform_2_tpl<float>(args);
        return;
    case kungfu::type_encoder::value<double>():
        _call_std_transform_2_tpl<double>(args);
        return;
    default:
        throw std::invalid_argument("std_transform_2 doesn't support dtype: " +
                                    std::to_string(dtype));
    }
}
