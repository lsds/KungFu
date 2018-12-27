#pragma once
#include <cstring>
#include <string>

#include <kungfu_types.hpp>
#include <libgo-kungfu.h>

inline GoSlice toGoSlice(const void *buffer, size_t count,
                         KungFu_Datatype dtype)
{
    const size_t size = kungfu_type_size(dtype) * count;
    return GoSlice{
        .data = (void *)(buffer),
        .len  = GoInt(size),
        .cap  = GoInt(size),
    };
}

inline GoString toGoString(const char *name)
{
    return GoString{
        .p = name,
        .n = (ptrdiff_t)(strlen(name)),
    };
}
