#include <stdml/bits/data/io.hpp>
#include <stdml/bits/data/state2.hpp>
#include <stdml/bits/data/tf_writer.hpp>
#include <stdml/elastic_state.hpp>

#include <fstream>
#include <iostream>
#include <tracer/patient>

namespace stdml::data
{
void save_shard_result(const std::string &filename, const shard_result &shard)
{
    std::ofstream f(filename);

    f << shard.filenames.size() << std::endl;
    for (const auto &filename : shard.filenames) {
        f << filename << std::endl;
    }

    f << shard.batch_sizes.size() << std::endl;
    for (const auto [bs, c] : shard.batch_sizes) {
        f << bs << ' ' << c << std::endl;
    }
}

std::string gen_filename(int64_t progress, int rank, int size, int idx)
{
    char filename[256];
    sprintf(filename, "from-%d-worker-%d-of-%d-%d.tf_record", (int)progress,
            rank, size, idx);
    return filename;
}

template <typename T>
std::vector<std::pair<T, int>> group(const std::vector<T> &xs)
{
    std::vector<std::pair<T, int>> g;
    if (xs.size() <= 0) {
        return g;
    }
    T x = xs[0];
    int c = 1;
    for (size_t i = 1; i < xs.size(); ++i) {
        if (xs[i] != x) {
            g.emplace_back(x, c);
            x = xs[i];
            c = 1;
        } else {
            ++c;
        }
    }
    g.emplace_back(x, c);
    return g;
}

shard_result write_tf_record(const ElasticState &es, state2 ds,
                             int global_batch_size, size_t max_sample_per_file)
{
    if (true) {
        fprintf(stderr, "%s ...\n", __func__);
    }
    WITH_PATIENT(__func__);

    std::vector<std::string> filenames;
    std::vector<int> batch_sizes;

    std::vector<std::string> buffer;

    auto write_buffer = [&]() {
        std::string filename =
            gen_filename(es.progress(), es.rank(), es.size(), filenames.size());
        std::ofstream f(filename, std::ios::binary);
        for (const auto &text : buffer) {
            f.write(text.data(), text.size());
        }
        filenames.push_back(filename);
        buffer.clear();
    };

    for (;;) {
        YIELD_PATIENT("...");
        auto [batch_idx, total] =
            ds.get_shard(es.rank(), es.size(), global_batch_size);

        batch_sizes.push_back(batch_idx.len());  // include 0-size batch
        if (batch_idx.len() == 0) {
            break;
        }

        auto batch = ds[batch_idx];
        // std::cout << "loaded " << batch.size() << " tf records" << std::endl;

        if (buffer.size() > 0 &&
            buffer.size() + batch.size() > max_sample_per_file) {
            write_buffer();
        }

        buffer.insert(buffer.end(), batch.begin(), batch.end());

        if (total < global_batch_size) {  // only possible in last batch
            break;
        }
    }

    if (buffer.size() > 0) {
        write_buffer();
    }

    fprintf(stderr, "%s created %d files for %s.\n", __func__,
            (int)filenames.size(), es.str().c_str());

    for (const auto [bs, c] : group(batch_sizes)) {
        std::cout << "batch size: " << bs << ", count: " << c << std::endl;
    }
    return shard_result{
        .filenames = filenames,
        .batch_sizes = group(batch_sizes),
    };
}
}  // namespace stdml::data
