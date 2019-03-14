#pragma once
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "error_checker.hpp"

struct show_cuda_error {
    std::string operator()(cudaError_t err) const
    {
        return std::to_string((int)err) + ": " + cudaGetErrorString(err);
    }
};

using cuda_checker = error_checker<cudaError_t, cudaSuccess, show_cuda_error>;

// cuda_stream wraps cudaStream_t
class cuda_stream
{
    cudaStream_t _stream;

  public:
    cuda_stream() { KUNGFU_CHECK(cuda_checker) << cudaStreamCreate(&_stream); }

    ~cuda_stream() { KUNGFU_CHECK(cuda_checker) << cudaStreamDestroy(_stream); }

    void sync()
    {
        KUNGFU_CHECK(cuda_checker) << cudaStreamSynchronize(_stream);
    }

    operator cudaStream_t() const { return _stream; }
};
