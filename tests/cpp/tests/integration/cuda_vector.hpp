#include <memory>

#include <cuda_runtime.h>

template <typename T> struct cuda_mem_allocator {
    T *operator()(int count)
    {
        T *deviceMem;
        cudaMalloc<T>(&deviceMem, count);
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { cudaFree(ptr); }
};

template <typename R> class cuda_vector
{
    const size_t count;
    const std::unique_ptr<R, cuda_mem_deleter> data_;

  public:
    explicit cuda_vector(size_t count)
        : count(count), data_(cuda_mem_allocator<R>()(count))
    {
    }

    R *data() { return data_.get(); }
};
