#include <proto/sample.pb.h>
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

void load_tf_record(const range_list_t &index, std::istream &is,
                    std::function<void(std::string)> f)
{
    load_tf_record(index, is, range_t(index.size()), f);
}

void load_tf_record(const range_list_t &index, std::istream &is,
                    const range_t r, std::function<void(std::string)> f)
{
    for (auto i : r) {
        f(std::move(read_region(index[i], is)));
    }
}

void load_tf_record(const range_list_t &index, std::istream &is,
                    const range_t r,
                    std::function<void(const tf_feature_map_wrap &)> f)
{
    for (auto i : r) {
        auto content = read_region(index[i], is);
        proto::Sample tf_file;
        if (!tf_file.ParseFromString(content)) {
            std::cerr << "not parsed!" << std::endl;
            continue;
        }
        tf_feature_map_wrap f_wrap(&tf_file.features().feature());
        f(f_wrap);
    }
}

void load_tf_record(const range_list_t &index, std::istream &is,
                    std::function<void(const tf_feature_map_wrap &)> f)
{
    load_tf_record(index, is, range_t(index.size()), f);
}
}  // namespace stdml::data
