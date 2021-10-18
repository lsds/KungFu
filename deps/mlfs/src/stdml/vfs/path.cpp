#include <stdml/bits/vfs/vfs.hpp>

namespace stdml::vfs
{
path::path(std::vector<int> parts) : super(std::move(parts))
{
}

bool path::isroot() const
{
    return empty();
}

path path::parent() const
{
    if (empty()) {
        throw std::runtime_error("root has no parent");
    }
    std::vector<int> pp(begin(), end() - 1);
    return path(std::move(pp));
}

int path::base() const
{
    if (empty()) {
        throw std::runtime_error("root has no base");
    }
    return *rbegin();
}
}  // namespace stdml::vfs
