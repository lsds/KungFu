#include <stdml/bits/data/summary.hpp>
#include <stdml/bits/utility/stat.hpp>

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <sstream>

std::string border(std::string text, std::string title);

namespace stdml::data
{
summary::summary()
{
    std::fill(counters_.begin(), counters_.end(), 0);
}

void summary::report(std::ostream &os) const
{
    os << get<file>() << " files";
    os << ", " << get<row>() << " rows";
    os << ", " << get<byte>() << " bytes";
    os << ' ' << '(' << show_size(get<byte>()) << ')';
    if (row_count_.size() > 0) {
        os << std::endl;
        for (const auto &[k, c] : row_count_) {
            os << c << ' ' << k << std::endl;
        }
    }
}

void summary::operator()(const std::string &key, int n)
{
    row_count_[key] += n;
}

void summary::done_file(int64_t n)
{
    add<file>(n);
}

void summary::done_row(int64_t n)
{
    add<row>(n);
}

void summary::done_bytes(int64_t n)
{
    add<byte>(n);
}

void summary::operator+=(const summary &s)
{
    std::transform(counters_.begin(), counters_.end(), s.counters_.begin(),
                   counters_.begin(), std::plus<int64_t>());
    for (const auto &[k, n] : s.row_count_) {
        (*this)(k, n);
    }
}

summary summary::operator+(const summary &s) const
{
    summary t = *this;
    t += s;
    return t;
}

std::ostream &operator<<(std::ostream &os, const summary &s)
{
    s.report(os);
    return os;
}

final_reporter::final_reporter(const summary &s, std::string name = "")
    : s(s), name_(std::move(name))
{
}

final_reporter::~final_reporter()
{
    std::stringstream ss;
    s.report(ss);
    std::cout << border(ss.str(), name_) + "\n";
}
}  // namespace stdml::data
