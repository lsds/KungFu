#pragma once
#include <ostream>
#include <string>
#include <vector>

#include <stdml/bits/data/index.hpp>
#include <stdml/bits/data/int_seq.hpp>

namespace stdml::data
{
class dataset_state
{
    std::string idx_file_;
    std::vector<std::string> filenames_;

    // can't be modified after init_once
    total_index index_;
    summary s_;
    using N = uint32_t;
    uint32_t seed_;

    //  TODO: moving state, move them a new class
    using Seq = int_seq<N>;

    int64_t progress_;
    Seq seq_;          // will be consumed every step
    Seq seq_predict_;  // will be updated on resize
    hasher hash_;

    friend std::ostream &operator<<(std::ostream &, const dataset_state &);

  public:
    dataset_state(std::string idx_file, std::vector<std::string> filenames);

    ~dataset_state();

    void save_index(std::string filename) const;

    int64_t len() const
    {
        return s_.rows();
    }

    N epoch_hash(uint32_t batch_size, uint32_t seed, int64_t progress) const;

    N epoch_hash(uint32_t batch_size, uint32_t seed) const;

    std::string operator[](int64_t i) const;

    std::vector<std::string> operator[](const Seq &batch_idx) const;

    // get hash
    N hash() const;

    int64_t progress() const;

    // TODO: support dynamic batch size
    bool validate_bash(uint32_t batch_size) const;

    // mutation operations

    void init(uint32_t seed);

    void sync(const std::function<N(N)> &broadcast, int rank, int size,
              int64_t progress, uint32_t batch_size);

    // update hash
    void hash(N h);

    void _seek(int64_t progress);

    Seq take(uint32_t bs);

    std::pair<Seq, int64_t> get_shard(const std::function<N(N)> &all_reduce_xor,
                                      int rank, int size, uint32_t bs);
};

std::ostream &operator<<(std::ostream &os, const dataset_state &ds);
}  // namespace stdml::data
