#include <stdml/bits/data/io.hpp>
#include <stdml/bits/data/tf_reader.hpp>

#include <fstream>
#include <iostream>

#include <ttl/nn/ops>

namespace stdml::data
{
range_list_t build_tf_index(std::string filename)
{
    std::ifstream is(filename);  // FIXME: std::ios::binary ?
    if (!is.is_open()) {
        throw std::runtime_error("failed to open: " + filename);
    }
    return build_tf_index(is);
}

range_list_t build_tf_index(std::istream &is)
{
    is.seekg(0, std::ios::end);
    const int64_t file_len = is.tellg();
    is.seekg(0, std::ios::beg);

    range_list_t index;
    int64_t off = 0;
    while (is.peek() != EOF) {
        struct rec_info {
            int64_t len;
            uint32_t crc1;
            uint32_t crc2;
        };
        static_assert(sizeof(rec_info) == 16, "");
        rec_info info;
        {
            is.read(reinterpret_cast<char *>(&info.len), sizeof(info.len));
            is.read(reinterpret_cast<char *>(&info.crc1), sizeof(info.crc1));
            is.ignore(info.len);
            is.read(reinterpret_cast<char *>(&info.crc2), sizeof(info.crc2));
        }
        index.emplace_back(
            region_t(off + sizeof(info.len) + sizeof(info.crc1), info.len));
        off += sizeof(rec_info) + info.len;
    }
    if (off != file_len) {
        throw std::runtime_error("unpexpected EOF after: " +
                                 std::to_string(off));
    }
    return index;
}
}  // namespace stdml::data
