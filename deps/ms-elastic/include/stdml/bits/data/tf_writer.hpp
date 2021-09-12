#pragma once
#include <functional>
#include <string>
#include <vector>

#include <stdml/bits/data/range.hpp>
#include <stdml/bits/data/state2.hpp>
#include <stdml/elastic_state.hpp>

namespace stdml::data
{
// returns a list of filenames
std::vector<std::string> write_tf_record(const ElasticState &es, state2 ds,
                                         int global_batch_size,
                                         size_t max_sample_per_file = 8192);
}  // namespace stdml::data
