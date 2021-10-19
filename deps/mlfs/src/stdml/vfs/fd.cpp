#include <stdml/bits/vfs/vfs.hpp>

namespace stdml::vfs
{
int fd_pool::get()
{
    for (int i = 3;; ++i) {
        if (fds_.count(i) == 0) {
            fds_[i] = i;
            return i;
        }
    }
}

void fd_pool::put(int fd)
{
    fds_.erase(fd);
}
}  // namespace stdml::vfs
