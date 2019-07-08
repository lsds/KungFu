#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "testing.hpp"

#include <kungfu/mst.hpp>

bool is_root = getSelfRank() == 0;

void test_AllReduce(kungfu_world &world, int np)
{
    TRACE_SCOPE(__func__);
    using T     = int32_t;
    const int n = np * 4;
    std::vector<T> x(n);
    std::vector<T> y(n);
    const auto dtype = kungfu::type_encoder::value<T>();

    std::iota(x.begin(), x.end(), 0);

    Waiter waiter;
    world.AllReduce(x.data(), y.data(), n, dtype, KungFu_SUM, "test-tensor",
                    [&waiter] { waiter.done(); });
    waiter.wait();

    int failed = 0;
    for (int i = 0; i < n; ++i) {
        const int expected = i * np;
        if (y[i] != expected) {
            printf("expected y[%d]=%d, but got %d\n", i, expected, y[i]);
            ++failed;
        }
    }
    if (failed) {
        printf("reduce %d elements among %d agents, %d elements failed\n", n,
               np, failed);
        exit(1);
    }
}

void bench_Reduce(kungfu_world &world, int n, int m)
{
    TRACE_SCOPE(__func__);

    std::vector<int32_t> x(n);
    std::vector<int32_t> y(n);
    const auto dtype = kungfu::type_encoder::value<int32_t>();
    const std::string name("fake_data");

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("KungfuReduce(async)");
        Waiter waiter;
        // void *recvBuf = is_root ? y.data() : nullptr;
        void *recvBuf = y.data();  // TODO: allow nullptr for non-root
        world.Reduce(x.data(), recvBuf, n, dtype, KungFu_SUM, name.c_str(),
                     [&waiter] { waiter.done(); });
        waiter.wait();
    }
}

void bench_AllReduce(kungfu_world &world, int n, int m)
{
    TRACE_SCOPE(__func__);

    std::vector<int32_t> x(n);
    std::vector<int32_t> y(n);
    const auto dtype = kungfu::type_encoder::value<int32_t>();
    const std::string name("fake_data");

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("KungfuAllReduce(async)");
        Waiter waiter;
        world.AllReduce(x.data(), y.data(), n, dtype, KungFu_SUM, name.c_str(),
                        [&waiter] { waiter.done(); });
        waiter.wait();
    }
}

void test_Gather(kungfu_world &world, int m)
{
    const int np   = world.ClusterSize();
    const int rank = getSelfRank();
    std::vector<int32_t> x(m);
    std::fill(x.begin(), x.end(), rank);
    if (is_root) {
        std::vector<int32_t> xs(m * np);
        world.Gather(x.data(), x.size(), KungFu_INT32,    //
                     xs.data(), xs.size(), KungFu_INT32,  //
                     "test-gather");
        const int sum = std::accumulate(xs.begin(), xs.end(), 0);
        if (sum != m * np * (np - 1) / 2) {
            printf("invalid Gather result\n");
            exit(1);
        }
    } else {
        world.Gather(x.data(), x.size(), KungFu_INT32,  //
                     nullptr, 0, KungFu_INT32,          //
                     "test-gather");
    }
}

template <typename T1, typename T2> class fake_transform
{
    void operator()(const T1 *input, int n1, T2 *output, int n2) const
    {
        for (int i = 0; i < n2; ++i) {
            output[i] = (i + 1) * std::accumulate(input, input + n1, 0);
        }
    }

  public:
    void operator()(const void *input, int input_count,
                    KungFu_Datatype input_dtype,  //
                    void *output, int output_count,
                    KungFu_Datatype output_dtype) const
    {
        if (kungfu::type_encoder::value<T1>() != input_dtype) {
            printf("invalid input_dtype");
            exit(1);
        }
        if (kungfu::type_encoder::value<T2>() != input_dtype) {
            printf("invalid output_dtype");
            exit(1);
        }
        (*this)(reinterpret_cast<const T1 *>(input), input_count,
                reinterpret_cast<T2 *>(output), output_count);
    }
};

void test_AllGatherTransform(kungfu_world &world)
{
    const int rank = getSelfRank();
    const int np   = getTestClusterSize();

    const int m = 10;
    const int n = 3;
    // input
    std::vector<int32_t> x(m);
    std::fill(x.begin(), x.end(), rank);
    std::vector<int32_t> y(n);
    std::fill(y.begin(), y.end(), 0);

    world.AllGatherTransform(x.data(), x.size(), KungFu_INT32,  //
                             y.data(), y.size(), KungFu_INT32,  //
                             "test-AllGatherTransform",
                             fake_transform<int32_t, int32_t>());
    for (int i = 0; i < n; ++i) {
        const int result = (i + 1) * m * np * (np - 1) / 2;
        if (y[i] != result) {
            printf("invalid AllGatherTransform result: %d, expect: %d\n", y[i],
                   result);
            exit(1);
        }
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    kungfu_world _kungfu_world;

    {
        const int np = getTestClusterSize();
        test_AllReduce(_kungfu_world, np);
    }
    {
        const int n = 100;
        const int m = 100;
        bench_Reduce(_kungfu_world, n, m);
    }
    {
        const int n = 100;
        const int m = 100;
        bench_AllReduce(_kungfu_world, n, m);
    }
    {
        const int m = 100;
        test_Gather(_kungfu_world, m);
    }
    {
        test_AllGatherTransform(_kungfu_world);
    }
    return 0;
}
