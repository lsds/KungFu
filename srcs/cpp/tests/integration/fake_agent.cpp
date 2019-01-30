#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "testing.hpp"

void test_sum(int np)
{
    TRACE_SCOPE(__func__);
    using T     = int32_t;
    const int n = np * 4;
    std::vector<T> x(n);
    std::vector<T> y(n);
    const auto dtype = kungfu::type_encoder::value<T>();

    std::iota(x.begin(), x.end(), 0);

    Waiter waiter;
    KungfuAllReduce(x.data(), y.data(), n, dtype, KungFu_SUM, "test-tensor",
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

void test(int n, int m)
{
    TRACE_SCOPE(__func__);

    std::vector<int32_t> x(n);
    std::vector<int32_t> y(n);
    const auto dtype = kungfu::type_encoder::value<int32_t>();
    std::string name("fake_data");

    for (int i = 0; i < m; ++i) {
        TRACE_SCOPE("KungfuAllReduceAsync");

        Waiter waiter;
        KungfuAllReduce(x.data(), y.data(), n, dtype, KungFu_SUM, name.c_str(),
                        [&waiter] { waiter.done(); });
        waiter.wait();
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    kungfu_world _kungfu_world;

    {
        const int np = getTestClusterSize();
        test_sum(np);
    }
    {
        const int n = 100;
        const int m = 100;
        test(n, m);
    }
    return 0;
}
