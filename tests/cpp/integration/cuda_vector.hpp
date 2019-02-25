#pragma once
#include <cstdio>
#include <memory>

#include <cuda_runtime.h>

#include "cuda_helper.hpp"

template <typename T> struct cuda_mem_allocator {
    T *operator()(int count)
    {
        T *deviceMem;
        // FIXME: make it work
        // CHECK(cuda_checker) << cudaMalloc<T>(&deviceMem, count);
        CHECK(cuda_checker) << cudaMalloc(&deviceMem, count * sizeof(T));
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { CHECK(cuda_checker) << cudaFree(ptr); }
};

template <typename T> class cuda_vector
{
    const size_t count;
    std::unique_ptr<T, cuda_mem_deleter> data_;

  public:
    explicit cuda_vector(size_t count)
        : count(count), data_(cuda_mem_allocator<T>()(count))
    {
    }

    T *data() { return data_.get(); }
};
