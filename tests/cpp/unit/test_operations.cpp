#include "testing.hpp"

template <typename T>
void test_allreduce(kungfu::Peer &kf, const int count)
{
    const auto dtype = kungfu::type_encoder::value<T>();
    std::vector<T> x(count);
    std::vector<T> y(count);

    std::iota(x.begin(), x.end(), 1);
    std::fill(y.begin(), y.end(), 0);

    kungfu::Waiter waiter;
    kf.AllReduce(x.data(), y.data(), count, dtype, KungFu_SUM, "test",
                 [&waiter] { waiter.done(); });
    waiter.wait();

    for (int i = 0; i < count; ++i) { ASSERT_EQ(y[i], i + 1); }
}

TEST(kungfu_AllReduce_test, test_global_step)
{
    kungfu::Peer kf;
    test_allreduce<int32_t>(kf, 10);
    test_allreduce<int32_t>(kf, 100);
}

template <typename T>
void test_allgather(kungfu::Peer &kf, const int count)
{
    const auto dtype    = kungfu::type_encoder::value<T>();
    const int count_all = count * kf.Size();
    std::vector<T> x(count);
    std::vector<T> y(count_all);

    std::iota(x.begin(), x.end(), 1);
    std::fill(y.begin(), y.end(), 0);

    kungfu::Waiter waiter;
    kf.AllGather(x.data(), count, dtype, y.data(), "test",
                 [&waiter] { waiter.done(); });
    waiter.wait();

    for (int i = 0; i < count_all; ++i) { ASSERT_EQ(y[i], i % count + 1); }
}

TEST(kungfu_AllGather_test, test_global_step)
{
    kungfu::Peer kf;
    test_allgather<int32_t>(kf, 10);
    test_allgather<int32_t>(kf, 100);
}
