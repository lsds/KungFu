#include <numeric>
#include <sstream>

#include <stdml/bits/data/index.hpp>
#include <stdml/bits/data/tf_reader.hpp>
#include <tracer/patient>
#include <tracer/site>

namespace stdml::data
{
total_index::total_index(std::vector<indexed_file> files)
    : files(std::move(files))
{
}

std::pair<int, int> total_index::file_index(int64_t i) const
{
    for (size_t fi = 0; fi < files.size(); ++fi) {
        if (static_cast<size_t>(i) < files[fi].index.size()) {
            return std::make_pair(fi, i);
        }
        i -= files[fi].index.size();
    }
    throw std::invalid_argument("row " + std::to_string(i) +
                                " out of file_index");
}

std::string total_index::operator[](int64_t i) const
{
    auto [fi, j] = TRACE_SITE_EXPR(file_index(i));
    std::ifstream fs;
    const auto &f = files.at(fi);
    TRACE_SITE_STMT(fs.open(f.filename));
    if (!fs.is_open()) {
        throw std::runtime_error("failed to open: " + f.filename);
    }
    const int j1 = j;
    return TRACE_SITE_EXPR(read_region(f.index.at(j1), fs));
}

summary total_index::stat() const
{
    stdml::execution::pmap_reduce_t<summary, indexed_file> pmap_reduce;
    return pmap_reduce(
        [](auto &f) {
            summary s;
            s.done_file();
            s.done_row(f.index.size());
            s.done_bytes(std::accumulate(f.index.begin(), f.index.end(),
                                         static_cast<int64_t>(0),
                                         [](int64_t acc, auto r) -> int64_t {
                                             return acc + r.len() + 16;
                                         }));
            return s;
        },
        files);
}

void total_index::save(std::ostream &os) const
{
    std::stringstream ss;
    ss << files.size() << std::endl;
    for (auto f : files) {
        ss << f.filename << ' ' << f.index.size() << std::endl;
        for (auto r : f.index) {
            ss << r.from << ' ' << r.to << std::endl;
        }
    }
    os << ss.str();
}

void total_index::save_with_meta(std::ostream &os) const
{
    std::stringstream ss;
    ss << files.size() << std::endl;
    for (auto f : files) {
        ss << f.filename << ' ' << f.index.size() << std::endl;
        for (auto r : f.index) {
            ss << r.from - 12 << ' ' << r.to + 4 << std::endl;
        }
    }
    os << ss.str();
}

void total_index::load(std::istream &os)
{
    // TRACE_SITE_SCOPE(__func__);

    int n;
    os >> n;
    files.clear();
    files.reserve(n);
    for (int i = 0; i < n; ++i) {
        indexed_file f;
        int m;
        os >> f.filename >> m;
        f.index.reserve(m);
        for (int j = 0; j < m; ++j) {
            int a, b;
            os >> a >> b;
            f.index.emplace_back(a, b);
        }
        files.emplace_back(std::move(f));
    }
}

total_index build_total_index(std::vector<std::string> filenames)
{
    WITH_PATIENT(__func__);
    int i = 0;
    std::vector<indexed_file> index;
    for (auto filename : filenames) {
        YIELD_PATIENT_("%d/%d", i++, (int)filenames.size());
        index.emplace_back(filename, build_tf_index(filename));
    }
    return total_index(std::move(index));
}

total_index load_total_index(std::string filename)
{
    total_index index;
    std::ifstream f(filename);
    index.load(f);
    return index;
}
}  // namespace stdml::data
