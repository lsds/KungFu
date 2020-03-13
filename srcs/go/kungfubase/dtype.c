#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "kungfu/dtype.h"

uint32_t kungfu_type_size(KungFu_Datatype dt)
{
#define CASE(t, T)                                                             \
    case t:                                                                    \
        return sizeof(T);

    switch (dt) {
        CASE(KungFu_UINT8, uint8_t);
        CASE(KungFu_UINT16, uint16_t);
        CASE(KungFu_UINT32, uint32_t);
        CASE(KungFu_UINT64, uint64_t);

        CASE(KungFu_INT8, int8_t);
        CASE(KungFu_INT16, int16_t);
        CASE(KungFu_INT32, int32_t);
        CASE(KungFu_INT64, int64_t);

        CASE(KungFu_FLOAT16, uint16_t);  //
        CASE(KungFu_FLOAT, float);
        CASE(KungFu_DOUBLE, double);

        CASE(KungFu_BOOL, char);
    default:
        fprintf(stderr, "unknown dtype: %d\n", (int)(dt));
        exit(1);
    };

#undef CASE
}
