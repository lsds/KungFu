#pragma once
#include <numeric>
#include <thread>
#include <vector>

#include <stdml/bits/data/range.hpp>

namespace stdml::execution
{
template <typename Ty, typename Tx>
struct pmap_t {
    template <typename F>
    auto operator()(const F &f, const std::vector<Tx> &xs,
                    const int m = std::thread::hardware_concurrency()) const
    {
        std::vector<Ty> ys(xs.size());
        std::vector<std::thread> ths;
        ths.reserve(m);
        for (auto k : data::range_t(m)) {
            ths.emplace_back([&, n = xs.size(), k = k] {
                for (size_t i = k; i < n; i += m) {
                    ys[i] = f(xs[i]);
                }
            });
        }
        for (auto &t : ths) {
            t.join();
        }
        return ys;
    }
};

template <typename Ty, typename Tx, typename F>
auto pmap(const F &f, const std::vector<Tx> &xs, const int m)
{
    return pmap_t<Ty, Tx>()(f, xs, m);
}

template <typename Ty, typename Tx, typename F>
auto pmap(const F &f, const std::vector<Tx> &xs)
{
    return pmap<Ty, Tx, F>(f, xs, std::thread::hardware_concurrency());
}

template <typename T, typename L>
auto sum(const L &xs)
{
    return std::accumulate(xs.begin(), xs.end(), T());
}

template <typename Ty, typename Tx>
struct pmap_reduce_t {
    template <typename F>
    auto operator()(const F &f, const std::vector<Tx> &xs,
                    const int m = std::thread::hardware_concurrency()) const
    {
        return sum<Ty>(pmap_t<Ty, Tx>()(f, xs, m));
    }
};
}  // namespace stdml::execution
