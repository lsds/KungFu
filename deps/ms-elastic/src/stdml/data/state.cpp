#include <stdexcept>

#include <stdml/bits/data/state.hpp>
#include <tracer/disable>
// #include <tracer/site>

namespace stdml::data
{
std::ostream &operator<<(std::ostream &os, const dataset_state &ds)
{
    os << ds.s_;
    return os;
}

dataset_state::dataset_state(std::string idx_file,
                             std::vector<std::string> filenames)
    : idx_file_(std::move(idx_file)), filenames_(std::move(filenames))
{
    if (!idx_file_.empty() && !filenames_.empty()) {
        throw std::invalid_argument(
            "idx_file and filename can't be both specified");
    }
    if (idx_file_.empty() && filenames_.empty()) {
        throw std::invalid_argument(
            "one of idx_file and filename must be specified");
    }
}

dataset_state::~dataset_state()
{
    // TODO: verify hash and progress
}

void dataset_state::save_index(std::string filename) const
{
    std::ofstream f(filename);
    index_.save(f);
}

dataset_state::N dataset_state::epoch_hash(uint32_t batch_size, uint32_t seed,
                                           int64_t progress) const
{
    Seq seq(s_.rows());
    seq.inplace_shuffle(seed);
    seq = seq.inplace_take(progress);  // FIXME: more efficient
    return seq.epoch_hash(batch_size);
}

dataset_state::N dataset_state::epoch_hash(uint32_t batch_size,
                                           uint32_t seed) const
{
    return epoch_hash(batch_size, seed, s_.rows());
}

std::string dataset_state::operator[](int64_t i) const
{
    return index_[i];
}

std::vector<std::string> dataset_state::operator[](const Seq &batch_idx) const
{
    std::vector<std::string> batch;
    batch.reserve(batch_idx.len());
    for (auto i : batch_idx.get()) {
        auto s = TRACE_SITE_EXPR(index_[i]);
        TRACE_SITE_STMT(batch.emplace_back(std::move(s)));
    }
    return batch;
}

dataset_state::N dataset_state::hash() const
{
    return static_cast<N>(hash_);
}

int64_t dataset_state::progress() const
{
    return progress_;
}

bool dataset_state::validate_bash(uint32_t batch_size) const
{
    return hash() == epoch_hash(batch_size, seed_, progress_);
}

void dataset_state::init(uint32_t seed)
{
    if (idx_file_.empty()) {
        fprintf(stderr, "building index for %d files\n",
                (int)filenames_.size());
        // FIXME: do this collectively
        index_ = build_total_index(filenames_);
    } else {
        // FIXME: do this collectively
        fprintf(stderr, "loading index from %s\n", idx_file_.c_str());
        std::ifstream f(idx_file_);
        index_.load(f);
    }
    s_ = index_.stat();
    // TODO: freeze index_ and s_ from now on

    seed_ = seed;
    progress_ = 0;
    seq_ = Seq(s_.rows());
    seq_.inplace_shuffle(seed_);
}

void dataset_state::sync(const std::function<N(N)> &broadcast, int rank,
                         int size, int64_t progress, uint32_t batch_size)
{
    TRACE_SITE_STMT(hash_.sync(broadcast));
    TRACE_SITE_STMT(_seek(progress));
    // pre_shard is slow when batch size is very small, e.g. bs=1
    seq_predict_ = TRACE_SITE_EXPR(seq_.fast_pre_shard(rank, size, batch_size));
    // TODO: restart prefetch
}

// update hash
void dataset_state::hash(N h)
{
    hash_(h);
}

void dataset_state::_seek(int64_t progress)
{
    if (progress_ == progress) {
        return;
    }
    progress_ = progress;
    seq_ = Seq(s_.rows());
    seq_.inplace_shuffle(seed_);
    seq_.inplace_drop(progress);
}

dataset_state::Seq dataset_state::take(uint32_t bs)
{
    auto s = seq_.inplace_take(bs);
    progress_ += s.len();
    return s;
}

std::pair<dataset_state::Seq, int64_t>
dataset_state::get_shard(const std::function<N(N)> &all_reduce_xor, int rank,
                         int size, uint32_t bs)
{
    auto batch_idx = take(bs);
    const int64_t total = batch_idx.len();  // maby be smaller than bs
    batch_idx = batch_idx.shard(rank, size);
    hash_(all_reduce_xor(batch_idx.batch_hash()));
    return std::make_pair(batch_idx, total);
}
}  // namespace stdml::data
