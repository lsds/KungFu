#include <stdml/bits/data/io.hpp>
#include <stdml/bits/data/range.hpp>

namespace stdml::data
{
std::string read_region(huge_region_t r, std::istream &fs)
{
    fs.seekg(r.off, std::ios::beg);
    std::string content;
    content.resize(r.len);
    if (fs.read(content.data(), r.len)) {
        return content;
    }
    throw std::runtime_error("read_region failed");
}

static huge_region_t stat_file(std::istream &fs)
{
    fs.seekg(0, std::ios::end);
    const int64_t file_len = fs.tellg();
    return huge_region_t(0, file_len);
}

std::string readfile(const std::string &filename)
{
    std::ifstream fs(filename);  // FIXME: std::ios::binary ?
    if (!fs.is_open()) {
        throw std::runtime_error("failed to open: " + filename);
    }
    auto reg = stat_file(fs);
    return read_region(reg, fs);
}
}  // namespace stdml::data
