#pragma once
#include <functional>
#include <string>
#include <vector>

#include <stdml/bits/data/range.hpp>
#include <stdml/bits/data/tf_feature.hpp>

namespace stdml::data
{
range_list_t build_tf_index(std::string filename);
range_list_t build_tf_index(std::istream &is);

void load_tf_record(const range_list_t &index, std::istream &is,
                    const range_t r, std::function<void(std::string)> f);

void load_tf_record(const range_list_t &index, std::istream &is,
                    std::function<void(std::string)> f);

void load_tf_record(const range_list_t &index, std::istream &is);

// load section
void load_tf_record(const range_list_t &index, std::istream &is,
                    const range_t r,
                    std::function<void(const tf_feature_map_wrap &)> f);

// load all
void load_tf_record(const range_list_t &index, std::istream &is,
                    std::function<void(const tf_feature_map_wrap &)> f);

}  // namespace stdml::data
