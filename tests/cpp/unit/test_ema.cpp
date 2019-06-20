#include "testing.hpp"
#include <ema.hpp>

TEST(kungfu_test, test_ema)
{
    using ema = ExponentialMovingAverage<>;
    ema g(.5);

    ASSERT_EQ(g.Get(), static_cast<ema::value_type>(0));

    g.Add(1);
    ASSERT_EQ(g.Get(), static_cast<ema::value_type>(1));

    g.Add(2);
    ASSERT_EQ(g.Get(), static_cast<ema::value_type>(1.5));
}
