#include <stdexcept>

#include <stdml/bits/data/state2.hpp>
#include <tracer/disable>
// #include <tracer/site>

namespace stdml::data
{
std::ostream &operator<<(std::ostream &os, const state2 &ds)
{
    os << ds.s_;
    return os;
}

template <typename N>
int_seq<N> build_seq(N n, uint32_t seed)
{
    int_seq<N> seq(n);
    seq.inplace_shuffle(seed);
    return seq;
}

state2::state2(std::string idx_file, uint32_t seed)
    : index_(load_total_index(std::move(idx_file))),
      seed_(seed),
      s_(index_.stat()),
      seq_(build_seq<N>(s_.rows(), seed_)),
      iter_(seq_)
{
}

state2::~state2()
{
    // TODO: verify hash and progress
}

std::vector<std::string> state2::operator[](const Seq &batch_idx) const
{
    std::vector<std::string> batch;
    batch.reserve(batch_idx.len());
    for (auto i : batch_idx.get()) {
        auto s = TRACE_SITE_EXPR(index_[i]);
        TRACE_SITE_STMT(batch.emplace_back(std::move(s)));
    }
    return batch;
}

int64_t state2::progress() const
{
    return iter_.tell();
}

void state2::sync(const std::function<N(N)> &broadcast, int64_t progress)
{
    // TRACE_SITE_STMT(hash_.sync(broadcast));
    iter_.seek(progress);

    // seq_predict_ = TRACE_SITE_EXPR(seq_.fast_pre_shard(rank, size,
    // batch_size));
    // TODO: restart prefetch
}

state2::Seq state2::_take(uint32_t bs)
{
    return iter_.take(bs);
}

std::pair<state2::Seq, int64_t> state2::get_shard(int rank, int size,
                                                  uint32_t bs)
{
    auto batch_idx = _take(bs);
    const int64_t total = batch_idx.len();  // maby be smaller than bs
    batch_idx = batch_idx.shard(rank, size);
    return std::make_pair(batch_idx, total);
}
}  // namespace stdml::data
