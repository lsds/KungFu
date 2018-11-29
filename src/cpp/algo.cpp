#include <cstdio>

#include <algorithm>
#include <functional>
#include <string>

#include <algo.h>
#include <mpi_types.hpp>

template <typename T>
void std_transform_2_tpl(const void *input_1, const void *input_2, void *output,
                         int n, int binary_op)
{
    std::transform(reinterpret_cast<const T *>(input_1),
                   reinterpret_cast<const T *>(input_1) + n,
                   reinterpret_cast<const T *>(input_2),
                   reinterpret_cast<T *>(output),
                   // FIXME: switch binary_op
                   std::plus<T>());
}

void std_transform_2(const void *input_1, const void *input_2, void *output,
                     int n, int dtype, int binary_op)
{
    switch (dtype) {
    case mpi::type_encoder::value<int>():
        std_transform_2_tpl<int>(input_1, input_2, output, n, binary_op);
        return;
    case mpi::type_encoder::value<float>():
        std_transform_2_tpl<float>(input_1, input_2, output, n, binary_op);
        return;
    default:
        throw std::invalid_argument("std_transform_2 doesn't support dtype: " +
                                    std::to_string(dtype));
    }
}
