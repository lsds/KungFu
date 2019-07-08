#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "testing.hpp"

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
            printf("invalid gather result\n");
            exit(1);
        }
    } else {
        world.Gather(x.data(), m, KungFu_INT32,  //
                     nullptr, 0, KungFu_INT32,   //
                     "test-gather");
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
    return 0;
}
