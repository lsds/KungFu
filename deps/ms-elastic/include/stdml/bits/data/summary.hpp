#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace stdml::data
{
class summary
{
    static constexpr int byte = 0;
    static constexpr int row = 1;
    static constexpr int file = 2;

    std::array<int64_t, 3> counters_;

    std::map<std::string, int64_t> row_count_;

    template <int i>
    void add(int64_t n)
    {
        std::get<i>(counters_) += n;
    }

    template <int i>
    int64_t get() const
    {
        return std::get<i>(counters_);
    }

  public:
    summary();

    void report(std::ostream &os = std::cout) const;

    void operator()(const std::string &key, int n = 1);
    void done_file(int64_t n = 1);
    void done_row(int64_t n = 1);
    void done_bytes(int64_t n = 1);

    int64_t rows() const
    {
        return get<row>();
    }

    int64_t bytes() const
    {
        return get<byte>();
    }

    void operator+=(const summary &s);

    summary operator+(const summary &s) const;
};

std::ostream &operator<<(std::ostream &, const summary &);

class final_reporter
{
    const summary &s;
    std::string name_;

  public:
    final_reporter(const summary &s, std::string name);

    ~final_reporter();
};
}  // namespace stdml::data

#define FINAL_REPORT(s, name)                                                  \
    stdml::data::final_reporter __final_reporter(s, name);
