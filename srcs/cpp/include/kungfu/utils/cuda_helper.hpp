#pragma once
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <kungfu/utils/error_checker.hpp>

struct show_cuda_error {
    std::string operator()(cudaError_t err) const
    {
        return std::to_string(static_cast<int>(err)) + ": " +
               cudaGetErrorString(err);
    }
};

using cuda_checker = error_checker<cudaError_t, cudaSuccess, show_cuda_error>;

// CudaStream wraps cudaStream_t
class CudaStream
{
    cudaStream_t stream_;

  public:
    CudaStream() { KUNGFU_CHECK(cuda_checker) << cudaStreamCreate(&stream_); }

    ~CudaStream()
    {
        const cudaError_t err = cudaStreamDestroy(stream_);
        if (err == cudaErrorCudartUnloading ||
            err == 29 /* driver shutting down */) {
            fprintf(stderr, "ignore cudaStreamDestroy error: %s\n",
                    show_cuda_error()(err).c_str());
            return;
        }
        KUNGFU_CHECK(cuda_checker) << err;
    }

    void sync()
    {
        KUNGFU_CHECK(cuda_checker) << cudaStreamSynchronize(stream_);
    }

    operator cudaStream_t() const { return stream_; }

    void memcpy(void *dst, const void *src, const size_t count,
                const cudaMemcpyKind dir)
    {
        KUNGFU_CHECK(cuda_checker)
            << cudaMemcpyAsync(dst, src, count, dir, stream_);
        sync();
    }
};
