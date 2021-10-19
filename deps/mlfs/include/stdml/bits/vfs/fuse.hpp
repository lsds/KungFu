#pragma once
#include <functional>
#include <stdml/bits/vfs/fuse-config.h>
#include <stdml/bits/vfs/vfs.hpp>

namespace stdml::vfs
{
class fuse
{
    static tree root;

  public:
    // int (*)(char const *, struct stat *);

    static int getattr(const char *path, struct stat *p);

    static int readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                       off_t offset, fuse_file_info *fi);

    static int open(char const *path, fuse_file_info *fi);

    static int read(const char *path, char *, size_t size, long off,
                    fuse_file_info *fi);

    static void run(const std::string &mount_path,
                    const std::function<void(tree &)> &init_fs,
                    const std::function<void()> &before_fuse_main,
                    bool test = false);
};
}  // namespace stdml::vfs
