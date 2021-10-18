#include <stdml/bits/data/state2.hpp>

namespace stdml::data
{
dataset_iter::dataset_iter(const Seq &seq, N pos) : seq_(seq), pos_(pos)
{
}

dataset_iter::N dataset_iter::tell() const
{
    return pos_;
}

void dataset_iter::seek(N pos)
{
    pos_ = pos;
}

void dataset_iter::operator+=(N d)
{
    pos_ += d;
}

dataset_iter::Seq dataset_iter::take(N bs)
{
    bs = std::min(bs, seq_.len() - pos_);
    basic_range_t<N> r(pos_, pos_ + bs);
    pos_ += bs;
    return seq_[r];
}
}  // namespace stdml::data
