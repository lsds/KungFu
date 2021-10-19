#pragma once
#include <fstream>

#include <stdml/bits/data/range.hpp>

namespace stdml::data
{
void read_region(huge_region_t r, std::istream &fs, void *buf);

std::string read_region(huge_region_t r, std::istream &fs);

std::string readfile(const std::string &filename);
}  // namespace stdml::data
