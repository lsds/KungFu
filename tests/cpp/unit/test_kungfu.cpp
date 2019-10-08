#include "testing.hpp"

TEST(kungfu_test, test_type_size)
{
    ASSERT_EQ(kungfu_type_size(KungFu_INT32), static_cast<uint32_t>(4));
    ASSERT_EQ(kungfu_type_size(KungFu_FLOAT16), static_cast<uint32_t>(2));
    ASSERT_EQ(kungfu_type_size(KungFu_FLOAT), static_cast<uint32_t>(4));
    ASSERT_EQ(kungfu_type_size(KungFu_DOUBLE), static_cast<uint32_t>(8));
}

TEST(kungfu_test, test_transform)
{
    int n = 1;
    std::vector<float> x(n);
    std::vector<float> y(n);
    x[0] = 1;
    y[0] = 2;
    std_transform_2(x.data(), y.data(), x.data(), n, KungFu_FLOAT, KungFu_SUM);
    ASSERT_FLOAT_EQ(x[0], static_cast<float>(3));
}
