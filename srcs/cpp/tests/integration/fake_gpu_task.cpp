#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "testing.hpp"

class check_cuda
{
  public:
    check_cuda(const std::string &file, int line)
        : _file(file), _line(line), _count(0)
    {
    }

    check_cuda &operator<<(cudaError_t error)
    {
        ++_count;
        if (error != cudaSuccess) {
            fprintf(stderr, "%d-th check at %s:%d failed, want %d, got %d\n",
                    _count, _file.c_str(), _line, cudaSuccess, error);
            exit(1);
        }
        return *this;
    }

  private:
    const std::string _file;
    const int _line;

    int _count;
};

#define CUDA_CHECK() check_cuda(__FILE__, __LINE__)

template <typename T>
bool differs(const std::vector<T> &x, const std::vector<T> &y)
{
    const int n   = x.size();
    bool has_diff = false;
    for (int i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            fprintf(stderr, "x[%d] = %s, but y[%d] = %s\n",  //
                    i, std::to_string(x[i]).c_str(),         //
                    i, std::to_string(y[i]).c_str());
            has_diff = true;
        }
    }
    return has_diff;
}

template <typename T> void test(kungfu_world &kf, int n, int m)
{
    TRACE_SCOPE(__func__);
    const auto dtype      = kungfu::type_encoder::value<T>();
    const int buffer_size = n * sizeof(T);
    const std::string name("fake_data");

    std::vector<T> x(n);
    std::vector<T> y(n);

    std::iota(x.begin(), x.end(), 1);

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("test GroupNegotiateGPUAsync once");

        const auto gs = kf.AdvanceGlobalStep();
        printf("[begin] global step #%d\n", gs);
        kf.SetGradientCount(1);

        int32_t *px;
        int32_t *py;
        CUDA_CHECK() << cudaMalloc((void **)&px, buffer_size)
                     << cudaMalloc((void **)&py, buffer_size);
        CUDA_CHECK() << cudaMemcpy(px, x.data(), buffer_size,
                                   cudaMemcpyHostToDevice);

        Waiter waiter;
        // TODO: test GroupNegotiateGPUAsync
        kf.NegotiateGPUAsync(px, py, n, dtype, KungFu_SUM, name.c_str(),
                             [&waiter] { waiter.done(); });
        waiter.wait();

        CUDA_CHECK() << cudaMemcpy(y.data(), py, buffer_size,
                                   cudaMemcpyDeviceToHost);
        CUDA_CHECK() << cudaFree(px) << cudaFree(py);

        // TODO: test multi-tasks
        if (differs(x, y)) {
            fprintf(stderr, "FAILED\n");
            exit(1);
        }
        printf("[end] global step #%d\n", gs);
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    kungfu_world _kungfu_world;
    const int n = 100;
    const int m = 10;
    test<int32_t>(_kungfu_world, n, m);
    return 0;
}
