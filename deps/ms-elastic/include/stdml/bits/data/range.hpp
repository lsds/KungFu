#pragma once
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace stdml::data
{
template <typename N>
struct basic_range_t;

template <typename N>
struct basic_region_t;

template <typename N>
struct basic_range_t {

    class iterator_t
    {
        N pos_;

      public:
        explicit iterator_t(N pos) : pos_(pos)
        {
        }

        bool operator!=(const iterator_t &it) const
        {
            return pos_ != it.pos_;
        }

        N operator*() const
        {
            return pos_;
        }

        void operator++()
        {
            ++pos_;
        }
    };

    N from;
    N to;

    template <typename M>
    basic_range_t(basic_range_t<M> r)
        : from(static_cast<N>(r.from)), to(static_cast<N>(r.to))
    {
    }

    basic_range_t() : basic_range_t(1)
    {
    }
    basic_range_t(N n) : basic_range_t(0, n)
    {
    }

    basic_range_t(N a, N b) : from(a), to(b)
    {
        if (a > b) {
            throw std::invalid_argument("invalid range: " + std::to_string(a) +
                                        ", " + std::to_string(b));
        }
    }

    template <typename M>
    basic_range_t(basic_region_t<M> r) : from(r.off), to(r.off + r.len)
    {
    }

    iterator_t begin() const
    {
        return iterator_t(from);
    }

    iterator_t end() const
    {
        return iterator_t(to);
    }

    N len() const
    {
        return to - from;
    }

    static N ceil_div(N n, N m)
    {
        return n % m ? n / m + 1 : n / m;
    }

    basic_range_t shard(N i, N n) const
    {
        const N k = ceil_div(len(), n);
        N a = i * k;
        N b = std::min<N>(a + k, len());
        if (a >= b) {
            return basic_range_t(to, to);
        }
        return basic_range_t(from + a, from + b);
    }

    static std::pair<N, N> divide(N n, N m)
    {
        N q = n / m;
        return std::make_pair(q, n - m * q);
    }

    basic_range_t even_shard(N i, N n) const
    {
        const auto [q, r] = divide(len(), n);
        if (i < r) {
            return basic_region_t(from + (q + 1) * i, q + 1);
        } else {
            return basic_region_t(from + (q + 1) * r + q * (i - r), q);
        }
    }
};

template <typename N>
std::ostream &operator<<(std::ostream &os, const basic_range_t<N> &r)
{
    os << '[' << r.from << ", " << r.to << ')';
    return os;
}

template <typename N>
struct basic_region_t {
    N off;
    N len;

    basic_region_t(N n) : off(0), len(n)
    {
    }

    basic_region_t(N a, N b) : off(a), len(b)
    {
    }

    template <typename M>
    basic_region_t(basic_range_t<M> r) : off(r.from), len(r.to - r.from)
    {
    }
};

using range_t = basic_range_t<uint32_t>;
using huge_range_t = basic_range_t<uint64_t>;

using region_t = basic_region_t<uint32_t>;
using huge_region_t = basic_region_t<uint64_t>;

using range_list_t = std::vector<range_t>;
}  // namespace stdml::data
