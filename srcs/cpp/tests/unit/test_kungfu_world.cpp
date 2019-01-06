#include "testing.hpp"

TEST(kungfu_world_test, test_construct)
{
    // FIXME: make it re-constructable
    // panic: Reuse of exported var name: total_msg_sent
    // kungfu_world kf;
}

TEST(kungfu_world_test, test_global_step)
{
    kungfu_world kf;
    for (int i = 0; i < 3; ++i) {
        int gs = kf.AdvanceGlobalStep();
        ASSERT_EQ(gs, i + 1);
    }
}
