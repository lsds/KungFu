#include "testing.hpp"

TEST(kungfu_world_test, test_construct)
{
    {
        kungfu_world kf1;
    }
    {
        kungfu_world kf2;
    }
}

TEST(kungfu_world_test, test_global_step)
{
    kungfu_world kf;
    for (int i = 0; i < 3; ++i) {
        int gs = kf.AdvanceGlobalStep();
        ASSERT_EQ(gs, i + 1);
    }
}
