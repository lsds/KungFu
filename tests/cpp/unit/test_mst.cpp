#include <numeric>
#include <random>
#include <vector>

#include "testing.hpp"

#include <kungfu/mst.hpp>

void gen_random_permu(std::vector<int> &p)
{
    std::iota(p.begin(), p.end(), 0);
    const int n = p.size();
    for (int i = 0; i < 10; ++i) {
        const int u = rand() % n;
        const int v = rand() % n;
        if (u != v) { std::swap(p[u], p[v]); }
    }
}

template <typename W> void gen_simple_graph(int n, std::vector<W> &weights)
{
    const auto set_w = [&](int i, int j, const W &w) {
        weights[i * n + j] = w;
    };

    std::vector<int> p(n);
    gen_random_permu(p);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                set_w(i, j, 0);
            } else {
                set_w(i, j, std::abs(p[i] - p[j]));
            }
        }
    }
}

TEST(kungfu_mst_test, test_mst)
{
    using W = int;
    using V = int;
    kungfu::MinimumSpanningTree<W, V> mst;

    std::vector<int> sizes(10);
    std::iota(sizes.begin(), sizes.end(), 1);
    for (const auto n : sizes) {
        std::vector<W> weights(n * n);
        std::vector<V> edges((n - 1) * 2);
        for (int i = 0; i < 3; ++i) {
            gen_simple_graph(n, weights);
            const W tot = mst(n, weights.data(), edges.data());
            ASSERT_EQ(tot, n - 1);
        }
    }
}
