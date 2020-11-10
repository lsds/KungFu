#include "testing.hpp"

#include <kungfu/numa/placement.hpp>

TEST(kungfu_placement_test, test_placement)
{
    using vec = std::vector<int>;
    {
        const vec cpus({9, 8, 7, 6, 5, 4, 3, 2});
        {
            const auto selected = kungfu::select_cpus(cpus, 3, 0, 5);
            ASSERT_EQ(selected, vec({9, 8, 7}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 3, 1, 5);
            ASSERT_EQ(selected, vec({9, 8, 7}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 3, 2, 5);
            ASSERT_EQ(selected, vec({6, 5, 4}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 3, 3, 5);
            ASSERT_EQ(selected, vec({6, 5, 4}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 3, 4, 5);
            ASSERT_EQ(selected, vec({3, 2}));
        }
    }
    {
        const vec cpus({0, 1, 2, 3, 4, 5, 6, 7});
        {
            const auto selected = kungfu::select_cpus(cpus, 2, 0, 4);
            ASSERT_EQ(selected, vec({0, 1, 2, 3}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 2, 1, 4);
            ASSERT_EQ(selected, vec({0, 1, 2, 3}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 2, 2, 4);
            ASSERT_EQ(selected, vec({4, 5, 6, 7}));
        }
        {
            const auto selected = kungfu::select_cpus(cpus, 2, 3, 4);
            ASSERT_EQ(selected, vec({4, 5, 6, 7}));
        }
    }
}
