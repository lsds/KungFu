#pragma once
#include <algorithm>
#include <array>

#include <ttl/bits/std_def.hpp>

namespace stdml
{
const char *show_size(int64_t n);

template <ttl::internal::rank_t r, typename N = int64_t>
struct basic_counters {
    std::array<N, r> counters_;

    basic_counters()
    {
        std::fill(counters_.begin(), counters_.end(), 0);
    }

    template <ttl::internal::rank_t i>
    void add(N n)
    {
        std::get<i>(counters_) += n;
    }

    template <ttl::internal::rank_t i>
    N get() const
    {
        return std::get<i>(counters_);
    }

    void add(const basic_counters &c)
    {
        std::transform(counters_.begin(), counters_.end(), c.counters_.begin(),
                       counters_.begin(), std::plus<N>());
    }

    basic_counters operator+(const basic_counters &c) const
    {
        basic_counters d = *this;
        d.add(d);
        return d;
    }
};
}  // namespace stdml

/*
#include <stdml/bits/tensor/dtype.hpp>
#include <stdml/bits/tensor/shape.hpp>

namespace stdml
{
struct model_info : public basic_counters<3> {
    static constexpr int params = 0;
    static constexpr int scalars = 1;
    static constexpr int bytes = 2;

    void operator()(const Shape &s, const DType &dt);
};

std::ostream &operator<<(std::ostream &os, const model_info &);
}  // namespace stdml
*/
