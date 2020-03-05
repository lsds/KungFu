#include <algorithm>
#include <numeric>
#include <vector>

#include "testing.hpp"

TEST(kungfu_order_group_test, test_order_group_simple)
{
    order_group_t *og = new_ranked_order_group(2);

    int idx = 0;
    int t0_exec_rank;
    int t1_exec_rank;

    order_group_do_rank(og, 1, new CallbackWrapper([&t1_exec_rank, &idx] {
                            printf("do 1\n");
                            t1_exec_rank = idx++;
                            printf("t1_exec_rank=%d\n", t1_exec_rank);
                        }));
    order_group_do_rank(og, 0, new CallbackWrapper([&t0_exec_rank, &idx] {
                            printf("do 0\n");
                            t0_exec_rank = idx++;
                            printf("t0_exec_rank=%d\n", t0_exec_rank);
                        }));

    std::vector<int32_t> arrive_order(2);
    order_group_wait(og, arrive_order.data());

    ASSERT_EQ(t0_exec_rank, 0);
    ASSERT_EQ(t1_exec_rank, 1);

    ASSERT_EQ(arrive_order[0], 1);
    ASSERT_EQ(arrive_order[1], 0);

    del_order_group(og);
}

void test_idempotent_wait(order_group_t *og)
{
    order_group_wait(og, nullptr);
    order_group_wait(og, nullptr);
}

void test_order_group(int n)
{
    order_group_t *og = new_ranked_order_group(n);

    std::vector<int> exec_order(n);
    std::fill(exec_order.begin(), exec_order.end(), -1);

    std::vector<int> arrive_order(n);
    std::iota(arrive_order.begin(), arrive_order.end(), 0);
    // TODO: random permutation
    std::reverse(arrive_order.begin(), arrive_order.end());

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        const int rank = arrive_order[i];
        order_group_do_rank(
            og, rank, new CallbackWrapper([&exec_order, rank = rank, &idx] {
                // printf("exec %d/%d\n", rank, n);
                exec_order[rank] = idx++;
            }));
    }

    order_group_wait(og, nullptr);  // TODO: test arrive order
    test_idempotent_wait(og);

    for (int i = 0; i < n; ++i) { ASSERT_EQ(exec_order[i], i); }

    del_order_group(og);
}

TEST(kungfu_order_group_test, test_order_group_comprehensive)
{
    test_order_group(10);
    test_order_group(100);
    test_order_group(200);
    test_order_group(1000);
}
