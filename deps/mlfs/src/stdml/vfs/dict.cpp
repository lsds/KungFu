#include <stdml/bits/vfs/vfs.hpp>

namespace stdml::vfs
{
int dict::operator[](const K &k)
{
    if (ids.count(k) > 0) {
        return ids.at(k);
    }
    int v = ids.size();
    ids[k] = v;
    words.emplace_back(k);
    return v;
}

dict::K dict::idx(int i) const
{
    return words.at(i);
}
}  // namespace stdml::vfs
