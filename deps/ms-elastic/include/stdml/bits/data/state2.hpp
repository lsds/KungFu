#pragma once
#include <ostream>
#include <string>
#include <vector>

#include <stdml/bits/data/index.hpp>
#include <stdml/bits/data/int_seq.hpp>

namespace stdml::data
{
class dataset_iter
{
    using N = uint32_t;
    using Seq = int_seq<N>;

    const Seq &seq_;
    N pos_;

  public:
    dataset_iter(const Seq &seq, N pos = 0);

    N tell() const;

    void seek(N pos);

    void operator+=(N d);

    Seq take(N bs);
};

class state2
{
    const total_index index_;
    const uint32_t seed_;
    const summary s_;

    using N = uint32_t;

    //  TODO: moving state, move them a new class
    using Seq = int_seq<N>;
    const Seq seq_;  // will be consumed every step

    dataset_iter iter_;

    // Seq seq_predict_;  // will be updated on resize

    friend std::ostream &operator<<(std::ostream &, const state2 &);

    Seq _take(uint32_t bs);

  public:
    state2(std::string idx_file, uint32_t seed);

    ~state2();

    int64_t len() const
    {
        return s_.rows();
    }

    std::vector<std::string> operator[](const Seq &batch_idx) const;

    int64_t progress() const;

    void sync(const std::function<N(N)> &broadcast, int64_t progress);

    std::pair<Seq, int64_t> get_shard(int rank, int size, uint32_t bs);
};

std::ostream &operator<<(std::ostream &os, const state2 &ds);
}  // namespace stdml::data
