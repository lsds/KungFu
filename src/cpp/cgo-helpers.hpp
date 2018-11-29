#pragma once
#include <cstring>
#include <string>

#include <go-kungfu.h>
#include <mpi_types.hpp>

inline GoSlice to_go_slice(const void *buffer, size_t count, int dtype)
{
    const size_t size = mpi::type_encoder::size(dtype) * count;
    return GoSlice{
        .data = (void *)(buffer),
        .len = GoInt(size),
        .cap = GoInt(size),
    };
}

inline GoString to_go_string(const char *name)
{
    return GoString{
        .p = name,
        .n = (ptrdiff_t)(strlen(name)),
    };
}
