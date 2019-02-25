#pragma once
#include <cstdio>
#include <memory>

#include <cuda_runtime.h>

#include "cuda_helper.hpp"

template <typename T> struct cuda_mem_allocator {
    T *operator()(int count)
    {
        T *deviceMem;
        CHECK(cuda_checker) << cudaMalloc<T>(&deviceMem, count);
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { CHECK(cuda_checker) << cudaFree(ptr); }
};

template <typename R> class cuda_vector
{
    const size_t count;
    std::unique_ptr<R, cuda_mem_deleter> data_;

  public:
    explicit cuda_vector(size_t count)
        : count(count), data_(cuda_mem_allocator<R>()(count))
    {
        printf("create cuda vector of size: %d\n", (int)sizeof(R) * count);
    }

    R *data() { return data_.get(); }
};
