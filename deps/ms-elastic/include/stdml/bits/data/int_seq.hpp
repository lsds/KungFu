#pragma once
#include <algorithm>
#include <iostream>

#include <stdml/bits/data/hash.hpp>
#include <stdml/bits/data/range.hpp>

namespace stdml::data
{
template <typename N>
class int_seq
{
    using S = std::vector<N>;

    S seq_;

  public:
    int_seq(N n = 0) : seq_(n)
    {
        std::iota(seq_.begin(), seq_.end(), 0);
    }

    int_seq(S seq) : seq_(std::move(seq))
    {
    }

    N len() const
    {
        return seq_.size();
    }

    const S &get() const
    {
        return seq_;
    }

    void inplace_shuffle(uint32_t seed)
    {
        std::mt19937 g(seed);
        // result is different on linux and osx
        std::shuffle(seq_.begin(), seq_.end(), g);
    }

    void inplace_shuffle()
    {
        fprintf(stderr, "[Warning] shuffle without seed!!!\n");
        std::random_device rd;
        inplace_shuffle(rd());
    }

    int_seq operator[](const basic_range_t<N> &r) const
    {
        if (r.to > seq_.size()) {
            throw std::invalid_argument("out of range");
        }
        S s(r.len());
        std::copy(seq_.begin() + r.from, seq_.begin() + r.to, s.begin());
        return s;
    }

    int_seq inplace_take(N n)
    {
        if (n >= seq_.size()) {
            S s = std::move(seq_);
            seq_.clear();
            return s;
        } else {
            S s(n);
            std::copy(seq_.begin(), seq_.begin() + n, s.begin());
            seq_.erase(seq_.begin(), seq_.begin() + n);
            return s;
        }
    }

    void inplace_drop(N n)
    {
        if (n >= seq_.size()) {
            seq_.clear();
        } else {
            seq_.erase(seq_.begin(), seq_.begin() + n);
        }
    }

    static N ceil_div(N n, N m)
    {
        return n % m ? n / m + 1 : n / m;
    }

    int_seq shard(N i, N m) const
    {
        N k = ceil_div(seq_.size(), m);
        N a = i * k;
        N b = std::min<N>(a + k, seq_.size());
        if (a >= b) {
            return S();
        }
        S s(b - a);
        std::copy(seq_.begin() + a, seq_.begin() + b, s.begin());
        return s;
    }

    // i : rank, m: cluster size, n : batch size
    int_seq pre_shard(N i, N m, N n) const
    {
        S s;
        for (int_seq r = seq_; r.len() > 0;) {
            auto t = r.inplace_take(n).shard(i, m);
            s.insert(s.end(), t.seq_.begin(), t.seq_.end());
        }
        return s;
    }

    int_seq fast_pre_shard(N i, N m, N n) const
    {
        S s;
        for (N k = 0; k < seq_.size(); k += n) {
            basic_range_t<N> r(k, std::min<N>(k + n, seq_.size()));
            r = r.shard(i, m);
            s.insert(s.end(), seq_.begin() + r.from, seq_.begin() + r.to);
        }
        return s;
    }

    void save(std::ostream &os)
    {
        std::stringstream ss;
        ss << seq_.size() << std::endl;
        for (auto x : seq_) {
            ss << x << std::endl;
        }
        os << ss.str();
    }

    template <typename F>
    void map(const F &f) const
    {
        for (auto i : seq_) {
            f(i);
        }
    }

    N batch_hash(N i, N j) const
    {
        return std::accumulate(seq_.begin() + i, seq_.begin() + j,
                               static_cast<N>(0), std::bit_xor<N>());
    }

    N batch_hash() const
    {
        return batch_hash(0, seq_.size());
    }

    N epoch_hash(N n, N progress) const
    {
        static_assert(std::is_same<N, uint32_t>::value,
                      "epoch_hash only suports u32");
        hasher h;
        for (N i = 0; i < progress; i += n) {
            N j = std::min(i + n, static_cast<N>(seq_.size()));
            h(batch_hash(i, j));
        }
        return static_cast<N>(h);
    }

    N epoch_hash(N n) const
    {
        return epoch_hash(n, seq_.size());
    }
};
}  // namespace stdml::data
