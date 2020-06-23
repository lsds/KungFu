#include <kungfu/cuda/stream.hpp>

namespace kungfu
{
CudaStream::CudaStream()
{
    KUNGFU_CHECK(cuda_checker) << cudaStreamCreate(&stream_);
}

CudaStream::~CudaStream()
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

void CudaStream::sync()
{
    KUNGFU_CHECK(cuda_checker) << cudaStreamSynchronize(stream_);
}

void CudaStream::memcpy(void *dst, const void *src, const size_t count,
                        const cudaMemcpyKind dir)
{
    KUNGFU_CHECK(cuda_checker)
        << cudaMemcpyAsync(dst, src, count, dir, stream_);
    sync();
}

CudaStream StreamPool::Get()
{
    std::lock_guard<std::mutex> _lk(mu_);
    if (queue_.empty()) {
        auto s = queue_.front();
        queue_.pop();
        return s;
    }
    CudaStream s;
    return s;
}

void StreamPool::Put(CudaStream stream)
{
    std::lock_guard<std::mutex> _lk(mu_);
    queue_.push(stream);
}
}  // namespace kungfu
