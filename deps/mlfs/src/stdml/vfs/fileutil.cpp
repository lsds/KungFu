#include <stdml/bits/vfs/vfs.hpp>

namespace stdml::vfs
{
void fileutil::mkprefix(const std::string &p, tree &root)
{
    if (p.empty()) { return; }
    if (p[0] != '/') { return; }
    int n = p.size();
    for (int i = 1; i < n; ++i) {
        if (p[i] == '/') {
            std::string pp(p.data(), p.data() + i);
            if (!root.exists(pp)) { root.mkdir(pp); }
        }
    }
}
}  // namespace stdml::vfs
