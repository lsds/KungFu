#pragma once
#include <fstream>

#include <stdml/bits/data/range.hpp>

namespace stdml::data
{
std::string read_region(huge_region_t r, std::istream &fs);

std::string readfile(const std::string &filename);
}  // namespace stdml::data
