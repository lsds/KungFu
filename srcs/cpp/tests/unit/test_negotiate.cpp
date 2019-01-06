#include "testing.hpp"

TEST(kungfu_negotiate_test, test_global_step)
{
    using T          = int32_t;
    const int count  = 10;
    const auto dtype = kungfu::type_encoder::value<T>();
    std::vector<T> x(count);
    std::vector<T> y(count);

    std::iota(x.begin(), x.end(), 1);
    std::fill(y.begin(), y.end(), 0);

    kungfu_world kf;
    Waiter waiter;
    kf.NegotiateAsync(x.data(), y.data(), count, dtype, KungFu_SUM, "test",
                      [&waiter] { waiter.done(); });
    waiter.wait();

    for (int i = 0; i < count; ++i) { ASSERT_EQ(y[i], i + 1); }
}
