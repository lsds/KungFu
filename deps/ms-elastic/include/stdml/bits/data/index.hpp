#pragma once
#include <stdml/bits/data/io.hpp>
#include <stdml/bits/data/range.hpp>
#include <stdml/bits/data/summary.hpp>
#include <stdml/execution>

namespace stdml::data
{
struct file_region {
    std::string filename;
    range_t r;

    void drop_front(int n)
    {
        const auto [a, b] = r;
        r = range_t(a + n, b);
    }

    void drop_back(int n)
    {
        const auto [a, b] = r;
        r = range_t(a, b - n);
    }

    void read(void *buf)
    {
        std::ifstream fs(filename);
        read_region(r, fs, buf);
    }
};

struct indexed_file {
    std::string filename;
    range_list_t index;

    indexed_file() = default;

    indexed_file(std::string filename, range_list_t index)
        : filename(std::move(filename)), index(std::move(index))
    {
    }
};

class total_index
{
    std::vector<indexed_file> files_;

    // idx -> (file_idx, reg_idx)
    std::vector<std::pair<int, int>> ridx_;

    void build_ridx();

  public:
    total_index() = default;

    total_index(std::vector<indexed_file> files);

    std::pair<int, int> file_index(int64_t i) const;

    std::string operator[](int64_t i) const;

    size_t region_size(int64_t idx) const;

    file_region get_file_region(int64_t idx) const;

    summary stat() const;

    void save(std::ostream &os) const;  // deprecated

    // save [from, to) as [from - 12, to + 4), for meta data bytes
    void save_with_meta(std::ostream &os) const;

    void load(std::istream &os);
};

total_index build_total_index(std::vector<std::string> filenames);

total_index load_total_index(std::string filename);
}  // namespace stdml::data
