#include "op.h"

#include <algorithm>
#include <cstdint>
#include <functional>

#include "f16.h"

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

    template <typename T> void call_as(int n, op o) const
    {
        const T *x = reinterpret_cast<const T *>(input1);
        const T *y = reinterpret_cast<const T *>(input2);
        T *z       = reinterpret_cast<T *>(output);
        switch (o) {
        case sum:
            std::transform(x, x + n, y, z, std::plus<T>());
            break;
        case min:
            std::transform(x, x + n, y, z, std_min<T>());
            break;
        case max:
            std::transform(x, x + n, y, z, std_max<T>());
            break;
        case prod:
            std::transform(x, x + n, y, z, std::multiplies<T>());
            break;
        default:
            exit(1);
        }
    }

    void call_as_f16(int n, op o) const
    {
        switch (o) {
        case sum:
            float16_sum(output, input1, input2, n);
            break;
        default:
            exit(1);
        }
    }
};

void std_transform_2(const void *input1, const void *input2, void *output,
                     const int n, const dtype dt, const op o)
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
        CASE(u8, uint8_t);
        CASE(u16, uint16_t);
        CASE(u32, uint32_t);
        CASE(u64, uint64_t);

        CASE(i8, int8_t);
        CASE(i16, int16_t);
        CASE(i32, int32_t);
        CASE(i64, int64_t);

    case f16:
        w.call_as_f16(n, o);
        break;

        CASE(f32, float);
        CASE(f64, double);
    default:
        exit(1);
    };

#undef CASE
}
