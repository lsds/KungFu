#pragma once
#include <functional>
#include <string>
#include <vector>

#include <stdml/bits/data/range.hpp>
#include <stdml/bits/data/state2.hpp>
#include <stdml/elastic_state.hpp>

namespace stdml::data
{
struct shard_result {
    std::vector<std::string> filenames;

    // list of (batch_size, count)
    std::vector<std::pair<int, int>> batch_sizes;
};

void save_shard_result(const std::string &filename, const shard_result &shard);

shard_result write_tf_record(const ElasticState &es, state2 ds,
                             int global_batch_size,
                             size_t max_sample_per_file = 8192);
}  // namespace stdml::data
