#pragma once
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <cuda_runtime.h>

#include <kungfu/utils/error_checker.hpp>

namespace kungfu
{
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
    CudaStream();

    ~CudaStream();

    void sync();

    operator cudaStream_t() const { return stream_; }

    void memcpy(void *dst, const void *src, const size_t count,
                const cudaMemcpyKind dir);
};

class StreamPool
{
    std::mutex mu_;
    std::queue<std::unique_ptr<CudaStream>> queue_;

    void debug();

  public:
    // StreamPool();
    ~StreamPool();

    std::unique_ptr<CudaStream> Get();

    void Put(std::unique_ptr<CudaStream> stream);
};
}  // namespace kungfu
