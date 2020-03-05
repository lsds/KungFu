#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
|00000000|   c    |    b   |00001000|
|01234567|01234567|01234567|01234567|
type-category :
    - unsigned: 0
    - signed: 1
    - float: 2
    - bool: 3

type-bytes : 1 2 4 8
bits-per-byte : 8
*/

#define TYPE_CODE(c, b) ((c << 16) | (b << 8) | 8)

enum KungFu_Datatype {
    KungFu_UINT8  = TYPE_CODE(0, 1),
    KungFu_UINT16 = TYPE_CODE(0, 2),
    KungFu_UINT32 = TYPE_CODE(0, 4),
    KungFu_UINT64 = TYPE_CODE(0, 8),

    KungFu_INT8  = TYPE_CODE(1, 1),
    KungFu_INT16 = TYPE_CODE(1, 2),
    KungFu_INT32 = TYPE_CODE(1, 4),
    KungFu_INT64 = TYPE_CODE(1, 8),

    KungFu_FLOAT16 = TYPE_CODE(2, 2),
    KungFu_FLOAT   = TYPE_CODE(2, 4),
    KungFu_DOUBLE  = TYPE_CODE(2, 8),

    KungFu_BOOL = TYPE_CODE(3, 1),
};

typedef enum KungFu_Datatype KungFu_Datatype;

#undef TYPE_CODE

extern uint32_t kungfu_type_size(KungFu_Datatype dt);

#ifdef __cplusplus
}
#endif
