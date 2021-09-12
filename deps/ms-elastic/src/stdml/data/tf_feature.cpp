#include <iostream>
#include <proto/feature.pb.h>
#include <stdml/bits/data/tf_feature.hpp>
#include <ttl/bits/std_reflect.hpp>

namespace stdml::data
{
tf_feature_map_wrap::tf_feature_map_wrap(const void *ptr) : ptr_(ptr)
{
}

const void *tf_feature_map_wrap::ptr() const
{
    return ptr_;
}

tf_feature_wrap::tf_feature_wrap(const void *ptr) : ptr_(ptr)
{
}

const void *tf_feature_wrap::ptr() const
{
    return ptr_;
}

template <typename T, typename L>
std::vector<T> copy_values(const L &xs)
{
    std::vector<T> vs(xs.value_size());
    for (size_t i = 0; i < vs.size(); ++i) {
        vs[i] = xs.value(i);
    }
    return vs;
}

std::vector<std::string>
tf_get_t<std::vector<std::string>>::operator()(const tf_feature_wrap &fv) const
{
    const auto &f = *reinterpret_cast<const proto::Feature *>(fv.ptr());
    if (f.kind_case() != proto::Feature::KindCase::kBytesList) {
        throw std::invalid_argument("tf_get_t<std::string>(...)");
    }
    return copy_values<std::string>(f.bytes_list());
}

std::vector<int64_t>
tf_get_t<std::vector<int64_t>>::operator()(const tf_feature_wrap &fv) const
{
    const auto &f = *reinterpret_cast<const proto::Feature *>(fv.ptr());
    if (f.kind_case() != proto::Feature::KindCase::kInt64List) {
        throw std::invalid_argument("tf_get_t<std::vector<int64_t>>(...)");
    }
    return copy_values<int64_t>(f.int64_list());
}

std::vector<float>
tf_get_t<std::vector<float>>::operator()(const tf_feature_wrap &fv) const
{
    const auto &f = *reinterpret_cast<const proto::Feature *>(fv.ptr());
    if (f.kind_case() != proto::Feature::KindCase::kFloatList) {
        throw std::invalid_argument("tf_get_t<std::vector<float>>(...)");
    }
    return copy_values<float>(f.float_list());
}

template <typename T>
T get_tf_feature_map_t<T>::operator()(const void *ptr, const std::string &name)
{
    using FM = google::protobuf::Map<std::string, proto::Feature>;
    const FM &fm = *reinterpret_cast<const FM *>(ptr);
    return tf_get_t<T>()(tf_feature_wrap(&fm.at(name)));
}

template struct get_tf_feature_map_t<std::vector<std::string>>;
template struct get_tf_feature_map_t<std::vector<int64_t>>;
template struct get_tf_feature_map_t<std::vector<float>>;

template struct get_tf_feature_map_t<std::string>;
template struct get_tf_feature_map_t<int64_t>;
template struct get_tf_feature_map_t<float>;
}  // namespace stdml::data
