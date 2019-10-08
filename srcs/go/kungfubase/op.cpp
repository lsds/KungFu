
#include <algorithm>
#include <cstdint>
#include <functional>

#include "f16.h"
#include "kungfu/op.h"

template <typename T> struct std_min {
    T operator()(const T &x, const T &y) const { return std::min(x, y); }
};

template <typename T> struct std_max {
    T operator()(const T &x, const T &y) const { return std::max(x, y); }
};

struct workspace {
    const void *input1;
    const void *input2;
    void *output;

    template <typename T> void call_as(const int n, const KungFu_Op o) const
    {
        const T *x = reinterpret_cast<const T *>(input1);
        const T *y = reinterpret_cast<const T *>(input2);
        T *z       = reinterpret_cast<T *>(output);
        switch (o) {
        case KungFu_SUM:
            std::transform(x, x + n, y, z, std::plus<T>());
            break;
        case KungFu_MIN:
            std::transform(x, x + n, y, z, std_min<T>());
            break;
        case KungFu_MAX:
            std::transform(x, x + n, y, z, std_max<T>());
            break;
        case KungFu_PROD:
            std::transform(x, x + n, y, z, std::multiplies<T>());
            break;
        default:
            exit(1);
        }
    }

    void call_as_f16(const int n, const KungFu_Op o) const
    {
        switch (o) {
        case KungFu_SUM:
            float16_sum(output, input1, input2, n);
            break;
        default:
            exit(1);
        }
    }
};

void std_transform_2(const void *input1, const void *input2, void *output,
                     const int n, const KungFu_Datatype dt, const KungFu_Op o)
{
    const workspace w = {
        .input1 = input1,
        .input2 = input2,
        .output = output,
    };

#define CASE(t, T)                                                             \
    case t:                                                                    \
        w.call_as<T>(n, o);                                                    \
        break

    switch (dt) {
        CASE(KungFu_UINT8, uint8_t);
        CASE(KungFu_UINT16, uint16_t);
        CASE(KungFu_UINT32, uint32_t);
        CASE(KungFu_UINT64, uint64_t);

        CASE(KungFu_INT8, int8_t);
        CASE(KungFu_INT16, int16_t);
        CASE(KungFu_INT32, int32_t);
        CASE(KungFu_INT64, int64_t);

    case KungFu_FLOAT16:
        w.call_as_f16(n, o);
        break;

        CASE(KungFu_FLOAT, float);
        CASE(KungFu_DOUBLE, double);
    default:
        exit(1);
    };

#undef CASE
}
