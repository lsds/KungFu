#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
|01234567|01234567|01234567|01234567|
type-category :
    - unsigned: 0
    - signed: 1
    - float: 2

type-bytes : 1 2 4 8
bits-per-byte : 8
*/

#define TYPE_CODE(c, b) ((c << 16) | (b << 8) | 8)

typedef enum {
    u8  = TYPE_CODE(0, 1),
    u16 = TYPE_CODE(0, 2),
    u32 = TYPE_CODE(0, 4),
    u64 = TYPE_CODE(0, 8),

    i8  = TYPE_CODE(1, 1),
    i16 = TYPE_CODE(1, 2),
    i32 = TYPE_CODE(1, 4),
    i64 = TYPE_CODE(1, 8),

    f16 = TYPE_CODE(2, 2),
    f32 = TYPE_CODE(2, 4),
    f64 = TYPE_CODE(2, 8),
} dtype;

#undef TYPE_CODE

extern int dtype_size(dtype dt);

#ifdef __cplusplus
}
#endif
