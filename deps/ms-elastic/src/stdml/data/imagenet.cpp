#include <proto/sample.pb.h>
#include <stdml/bits/data/decoder.hpp>
#include <stdml/data/imagenet>
#include <tracer/patient>
#include <tracer/rate>
#include <ttl/range>

namespace stdml::data
{
const imagenet::s_col_t imagenet::format("image/format");
const imagenet::s_col_t imagenet::class_("image/class/text");
const imagenet::s_col_t imagenet::colorspace("image/colorspace");
const imagenet::s_col_t imagenet::image("image/encoded");
const imagenet::i64_col_t imagenet::height("image/height");
const imagenet::i64_col_t imagenet::width("image/width");
const imagenet::i64_col_t imagenet::label("image/class/label");

void imagenet::default_parser::operator()(TensorRef x, TensorRef y_,
                                          tf_feature_map_wrap f) const
{
    // TRACE_SITE_SCOPE_RATE("parse sample", "sample", 1);
    decode_resize_jpeg_hw3(image(f), x);
    y_.typed<uint32_t, 0>() = label(f);
}

void imagenet::default_parser::operator()(TensorRef x, TensorRef y_,
                                          std::string s) const
{
    proto::Sample tf_file;
    if (!tf_file.ParseFromString(s)) {
        std::cerr << "not parsed!" << std::endl;
        throw std::runtime_error("failed to parse tf_file in tf_record");
    }
    tf_feature_map_wrap f_wrap(&tf_file.features().feature());
    (*this)(x, y_, f_wrap);
}

std::pair<Tensor, Tensor>
imagenet::default_parser::operator()(tf_feature_map_wrap f) const
{
    Tensor x(u8, {224, 224, 3});
    Tensor y_(u32);
    decode_resize_jpeg_hw3(image(f), x);
    y_.typed<uint32_t, 0>() = label(f);
    // std::cout << info(x) << ' ' << info(y_) << std::endl;
    return std::make_pair(std::move(x), std::move(y_));
}

std::pair<Tensor, Tensor>
imagenet::default_parser::operator()(std::string s) const
{
    proto::Sample tf_file;
    if (!tf_file.ParseFromString(s)) {
        std::cerr << "not parsed!" << std::endl;
        throw std::runtime_error("failed to parse tf_file in tf_record");
    }
    tf_feature_map_wrap f_wrap(&tf_file.features().feature());
    return (*this)(f_wrap);
}

std::pair<Tensor, Tensor>
imagenet::default_parser::operator()(std::vector<std::string> tf_files) const
{
    WITH_PATIENT("imagenet::default_parser::operator()");
    const int n = tf_files.size();
    TRACE_SITE_SCOPE_RATE("parse batch", "sample", n);
    Tensor xs(u8, {n, 224, 224, 3});
    Tensor y_s(u32, {n});
    for (auto i : ttl::range(n)) {
        YIELD_PATIENT_("parsed %d/%d", i, n);
        (*this)(xs[i], y_s[i], std::move(tf_files[i]));
    }
    return std::make_pair(std::move(xs), std::move(y_s));
}
}  // namespace stdml::data
