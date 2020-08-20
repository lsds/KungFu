#pragma once
#include <cstdio>
#include <memory>

#include <cuda_runtime.h>

#include <kungfu/cuda/stream.hpp>

template <typename T> struct cuda_mem_allocator {
    T *operator()(int count)
    {
        T *deviceMem;
        // FIXME: make it work
        // KUNGFU_CHECK(cuda_checker) << cudaMalloc<T>(&deviceMem, count);
        KUNGFU_CHECK(cuda_checker) << cudaMalloc(&deviceMem, count * sizeof(T));
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { KUNGFU_CHECK(cuda_checker) << cudaFree(ptr); }
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

    void from_host(const T *buffer)
    {
        KUNGFU_CHECK(cuda_checker) << cudaMemcpy(
            data_.get(), buffer, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    void to_host(T *buffer)
    {
        KUNGFU_CHECK(cuda_checker) << cudaMemcpy(
            buffer, data_.get(), count * sizeof(T), cudaMemcpyDeviceToHost);
    }
};
