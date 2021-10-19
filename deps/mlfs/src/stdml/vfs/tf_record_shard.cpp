#include <algorithm>
#include <cinttypes>
#include <sstream>

#include <listutil.hpp>
#include <stdml/bits/data/state2.hpp>
#include <stdml/bits/vfs/logging.hpp>
#include <stdml/bits/vfs/tf_record_shard.hpp>
#include <stdml/bits/vfs/vfs.hpp>
#include <stdml/elastic_state.hpp>

#define LOG() stdml::vfs::logger::get("tf_record_shard.log")

namespace stdml::vfs
{
class tf_record_shard : public node, public file_node
{
    const data::total_index &index_;
    using N   = uint32_t;
    using Seq = data::int_seq<N>;
    Seq seq_;

    std::vector<uint32_t> lefts_;
    std::vector<uint32_t> rights_;
    std::vector<uint32_t> sizes_;  // sizes of each region
    int64_t size_;

    class range_reader : public file_reader
    {
        const tf_record_shard &shard_;
        size_t pos_;

      public:
        range_reader(const tf_record_shard &shard, size_t pos)
            : shard_(shard), pos_(pos)
        {
        }

        size_t remain() const override { return shard_.size_ - pos_; }

        // l0, l1, l2, l3, ... l_{r-1}
        // [), [), [), [), ... [)
        //         [pos, pos+n)
        size_t read(char *buf, size_t limit) override
        {
            const size_t n = std::min<size_t>(remain(), limit);
            // [pos_, pos_ + n)
            data::range_t r(pos_, pos_ + n);
            const auto &[a, b] = r;

            const auto &rr = shard_.rights_;
            int i = std::upper_bound(rr.begin(), rr.end(), a) - rr.begin();
            const auto &ll = shard_.lefts_;
            int j = std::lower_bound(ll.begin(), ll.end(), b) - ll.begin();

            LOG().fmt("read(%d + %d), %d regions", (int)pos_, (int)limit,
                      j - i);

            {  // read regions
                size_t off = 0;
                for (int k = i; k < j; ++k) {
                    const auto l = ll[k];
                    const auto r = rr[k];
                    auto fr =
                        shard_.index_.get_file_region(shard_.seq_.get()[k]);
                    if (l < a) {
                        fr.drop_front(a - l);
                        // LOG().fmt("drop_front(%d)", a - l);
                    }
                    if (r > b) {
                        fr.drop_back(r - b);
                        // LOG().fmt("drop_back(%d)", r - b);
                    }
                    fr.read(buf + off);
                    off += fr.r.len();
                }
                if (off != n) {
                    LOG().fmt(
                        "INVALID read(%d + %d), %d regions, want: %d, got: %d",
                        (int)pos_, (int)limit, j - i, n, off);
                }
            }
            pos_ += n;
            return n;
        }
    };

  public:
    tf_record_shard(const stdml::data::total_index &index, Seq seq)
        : index_(index), seq_(std::move(seq))
    {
        uint32_t off = 0;
        for (auto i : seq_.get()) {
            const uint32_t l = index_.region_size(i);
            sizes_.emplace_back(l);
            lefts_.emplace_back(off);
            off += l;
            rights_.emplace_back(off);
        }
        size_ = std::accumulate(sizes_.begin(), sizes_.end(),
                                static_cast<size_t>(0));
        // std::cout << "shard size: " << size_ << ", has " << sizes_.size()
        //           << " regions" << std::endl;
    }

    bool isdir() const { return false; }

    file_node *as_file() { return this; }

    dir_node *as_dir() { throw std::runtime_error("not a dir"); }

    std::unique_ptr<file_reader> openat(size_t off) override
    {
        range_reader *p = new range_reader(*this, off);
        return std::unique_ptr<file_reader>(p);
    }
};
}  // namespace stdml::vfs

namespace stdml::data
{
std::string cluster_prefix(int64_t progress, int size)
{
    char path[1 << 10];
    sprintf(path, "/progress-%" PRId64 "/cluster-of-%d", progress, size);
    return path;
}

std::string worker_prefix(int64_t progress, int rank, int size)
{
    char path[1 << 10];
    sprintf(path, "/rank-%03d", rank);
    return cluster_prefix(progress, size) + path;
}

std::string shard_file_name(int idx)
{
    char filename[16];
    sprintf(filename, "%04d.tf_record", idx);
    return filename;
}

struct meta {
    std::string filename;
    uint32_t records;
};

void mount_tf_record_shards(vfs::tree &root, const data::total_index &index,
                            const int64_t progress, const int cluster_size,
                            const int rank, state2 ds, int global_batch_size,
                            size_t max_sample_per_file)
{
    const auto prefix = worker_prefix(progress, rank, cluster_size);
    ds.sync(progress);

    std::vector<meta> metas;
    std::vector<int> batch_sizes;
    using N   = uint32_t;
    using Seq = int_seq<N>;
    Seq buffer;

    auto write_buffer = [&]() {
        const int idx        = metas.size();
        std::string filename = prefix + "/" + shard_file_name(idx);
        meta m               = {
            .filename = filename,
            .records  = buffer.len(),
        };
        metas.emplace_back(m);
        vfs::fileutil::mkprefix(filename, root);
        root.touch(filename + ".meta", [&] {
            std::stringstream ss;
            for (auto i : buffer.get()) { ss << i << std::endl; }
            return ss.str();
        }());
        root.touch(filename,
                   new vfs::tf_record_shard(index, std::move(buffer)));
    };

    bool drop   = true;
    int dropped = 0;

    for (;;) {
        // printf("taking batch (%d/%d)...\n", es.rank(), es.size());
        auto [batch_idx, total] =
            ds.get_shard(rank, cluster_size, global_batch_size);
        if (total < global_batch_size && drop) {
            dropped = total;
            break;
        }
        if (buffer.len() >= max_sample_per_file) { write_buffer(); }
        batch_sizes.push_back(batch_idx.len());  // include 0-size batch
        buffer += batch_idx;
        if (batch_idx.len() == 0) { break; }

        if (total < global_batch_size) {  // only possible in last batch
            break;
        }
    }

    if (buffer.len() > 0) { write_buffer(); }

    root.touch(prefix + "/list.txt", [&] {
        std::string text;
        for (const auto &meta : metas) { text += meta.filename + "\n"; }
        return text;
    }());
    root.touch(prefix + "/batch-sizes.txt", [&] {
        std::stringstream ss;
        for (const auto [bs, c] : group(batch_sizes)) {
            ss << bs << " " << c << std::endl;
        }
        return ss.str();
    }());
    root.touch(prefix + "/info.txt", [&] {
        std::stringstream ss;
        for (const auto &meta : metas) {
            ss << meta.filename << " " << meta.records << std::endl;
        }
        return ss.str();
    }());
    root.touch(prefix + "/dropped.txt", [&] {
        std::stringstream ss;
        ss << dropped << std::endl;
        return ss.str();
    }());
    LOG().fmt("%s done", __func__);
}

void mount_tf_index(vfs::tree &root, const data::total_index &index)
{
    root.touch("/index.txt", [&] {
        std::stringstream ss;
        index.save_with_meta(ss);
        return ss.str();
    }());
}

void mount_tf_record_for_progress(vfs::tree &root,
                                  const data::total_index &index,
                                  const int64_t progress,
                                  const int cluster_size, state2 ds,
                                  int global_batch_size,
                                  size_t max_sample_per_file)
{
    char name[1 << 8];
    sprintf(name, "progress-%" PRId64 "-cluster-of-%d.txt", progress,
            cluster_size);
    root.touch(name, [&] {
        std::stringstream ss;
        for (int i = 0; i < cluster_size; ++i) {
            ss << worker_prefix(progress, i, cluster_size) << std::endl;
        }
        return ss.str();
    }());
    for (int i = 0; i < cluster_size; ++i) {
        mount_tf_record_shards(root, index, progress, cluster_size, i, ds,
                               global_batch_size, max_sample_per_file);
    }
    root.touch(
        cluster_prefix(progress, cluster_size) + "/global-batch-size.txt", [&] {
            std::stringstream ss;
            ss << global_batch_size << std::endl;
            return ss.str();
        }());
}
}  // namespace stdml::data
