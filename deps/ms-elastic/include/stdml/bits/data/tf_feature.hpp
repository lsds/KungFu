#pragma once
#include <stdexcept>
#include <string>
#include <vector>

namespace stdml::data
{
class tf_feature_map_wrap
{
    const void *ptr_;

  public:
    tf_feature_map_wrap(const void *ptr);

    const void *ptr() const;
};

class tf_feature_wrap
{
    const void *ptr_;

  public:
    tf_feature_wrap(const void *ptr);

    const void *ptr() const;
};

template <typename T>
struct tf_get_t;

template <>
struct tf_get_t<std::vector<std::string>> {
    std::vector<std::string> operator()(const tf_feature_wrap &f) const;
};

template <>
struct tf_get_t<std::vector<int64_t>> {
    std::vector<int64_t> operator()(const tf_feature_wrap &f) const;
};

template <>
struct tf_get_t<std::vector<float>> {
    std::vector<float> operator()(const tf_feature_wrap &f) const;
};

template <typename T>
struct tf_get_t {
    T operator()(const tf_feature_wrap &f) const
    {
        using L = std::vector<T>;
        L vs = tf_get_t<L>()(f);
        if (vs.size() != 1) {
            throw std::runtime_error("tf_get_t<T>: tf_feature size != 1");
        }
        return std::move(vs[0]);
    }
};

template <typename T>
struct get_tf_feature_map_t {
    T operator()(const void *fm, const std::string &name);
};

template <typename T>
T tf_get(const tf_feature_map_wrap fm, const std::string &name)
{
    return get_tf_feature_map_t<T>()(fm.ptr(), name);
}

template <typename T>
class get_tf_col
{
    std::string name_;

  public:
    get_tf_col(std::string name) : name_(std::move(name))
    {
    }

    T operator()(const tf_feature_map_wrap &fm) const
    {
        return tf_get<T>(fm, name_);
    }
};
}  // namespace stdml::data
