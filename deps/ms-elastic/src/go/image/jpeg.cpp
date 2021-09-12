#include <stdml/bits/data/decoder.hpp>
#include <stdml/tensor>
#include <ttl/consistent_variable>
#include <vector>

#include "libjpeg-decoder-cgo.h"

namespace stdml::data
{
void decode_jpeg(const std::string &s)
{
    GoDecodeJpeg(const_cast<char *>(s.data()), s.size());
}
void decode_resize_jpeg_hw3(const std::string &s, TensorRef x)
{
    auto t = x.typed<uint8_t, 3>();
    int h, w;
    ttl::consistent_variable<int> c(3);
    std::tie(h, w, c) = std::tuple_cat(t.dims());
    if (int err = GoDecodeJpegHW3(const_cast<char *>(s.data()), s.size(), h, w,
                                  t.data());
        err > 0) {
        fprintf(stderr, "GoDecodeJpegHW3 failed\n");
    }
}

Tensor decode_resize_jpeg_hw3(const std::string &s, int h, int w)
{
    Tensor x(u8, {h, w, 3});
    decode_resize_jpeg_hw3(s, x);
    return x;
}
}  // namespace stdml::data
