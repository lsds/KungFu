#pragma once
#include <algorithm>
#include <vector>

#include <kungfu.h>

namespace kungfu
{
template <typename Weight, typename Vertex> class MinimumSpanningTree
{
    Weight prim(const int n, const Weight *weights, Vertex *edges,
                const int seed) const
    {
        const auto w = [&](int i, int j) {
            return (weights[i * n + j] + weights[j * n + i]) / 2;
        };
        const auto put_edge = [&](int i, int u, int v) {
            edges[i * 2 + 0] = u;
            edges[i * 2 + 1] = v;
        };

        std::vector<bool> used(n);
        std::fill(used.begin(), used.end(), false);
        used[seed] = true;

        std::vector<int> best(n);
        std::vector<int> from(n);
        for (int i = 0; i < n; ++i) {
            best[i] = w(seed, i);
            from[i] = seed;
        }

        Weight tot = 0;
        for (int i = 1; i < n; ++i) {
            int k = std::find(used.begin(), used.end(), false) - used.begin();
            for (int j = k + 1; j < n; ++j) {
                if (!used[j] && best[j] < best[k]) { k = j; }
            }
            used[k] = true;
            tot += best[k];
            put_edge(i - 1, from[k], k);
            for (int j = 0; j < n; ++j) {
                if (!used[j] && w(k, j) < best[j]) {
                    best[j] = w(k, j);
                    from[j] = k;
                }
            }
        }
        return tot;
    }

  public:
    Weight operator()(const int n, const Weight *weights, Vertex *edges,
                      const int seed = 0) const
    {
        return prim(n, weights, edges, seed);
    }
};
}  // namespace kungfu
