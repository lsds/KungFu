#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dtype.h"

int dtype_size(dtype dt)
{
#define CASE(t, T)                                                             \
    case t:                                                                    \
        return sizeof(T);

    switch (dt) {
        CASE(u8, uint8_t);
        CASE(u16, uint16_t);
        CASE(u32, uint32_t);
        CASE(u64, uint64_t);

        CASE(i8, int8_t);
        CASE(i16, int16_t);
        CASE(i32, int32_t);
        CASE(i64, int64_t);

        CASE(f16, uint16_t);  //
        CASE(f32, float);
        CASE(f64, double);
    default:
        fprintf(stderr, "unknown dtype: %d\n", (int)(dt));
        exit(1);
    };

#undef CASE
}
