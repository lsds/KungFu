#include <cstring>
#include <ctime>
#include <cxxabi.h>
#include <sstream>
// #include <format> // Not available

#include <stdml/bits/vfs/fuse.hpp>
#include <stdml/bits/vfs/logging.hpp>
#include <unistd.h>

constexpr const char *logfile = "mlfs-01.log";

template <typename T>
std::string demangled_type_info_name()
{
    int status = 0;
    return abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
}

#define SHOW_FN_PTR(fp, name, f)                                               \
    {                                                                          \
        using fn_t        = decltype(f);                                       \
        std::string tname = demangled_type_info_name<fn_t>();                  \
        fprintf(fp, "// %s :: %s;\n", #name, tname.c_str());                   \
    }

#define DEF_FUSE_FUNC(h, name, func)                                           \
    {                                                                          \
        /* SHOW_FN_PTR(stderr, name, h.name); */                               \
        /* fprintf(stderr, "using %s=%s @ %p\n", #name, #func, func); */       \
        /* logger::get().fmt("using %s=%s @ %p", #name, #func, func); */       \
        h.name = func;                                                         \
    }

#define LOG() logger::get(logfile)

namespace stdml::vfs
{
std::string show_hex(int x)
{
    char line[32];
    // sprintf(line, "0x%08x", x);
    if (x == S_IFREG) {
        sprintf(line, "0x%x (S_IFREG)", x);
    } else {
        sprintf(line, "0x%x", x);
    }
    return line;
}

// https://libfuse.github.io/doxygen/structfuse__file__info.html
std::string show_fuse_file_info(fuse_file_info *fi)
{
    std::stringstream ss;
    ss << "fuse_file_info<";
    ss << "fh=" << fi->fh;
    ss << ", flags=" << show_hex(fi->flags);
    ss << ">";
    return ss.str();
}

tree fuse::root;
fd_pool pool;

int64_t ceil_div(int64_t a, int64_t b) { return a % b ? a / b + 1 : a / b; }

int fuse::getattr(const char *path, struct stat *p)
{
    LOG().fmt("%s(%s, %p)", __func__, path, p);
    node *n = root.open(path);
    if (n == nullptr) {
        LOG().fmt("%s(%s, %p) :: not found", __func__, path, p);
        return -1;
    }

    p->st_dev = 0;
    p->st_ino = 0;

    p->st_uid = getuid();
    p->st_gid = getgid();

    mode_t read_perm = S_IRUSR | S_IRGRP;
    mode_t exe_perm  = S_IXUSR | S_IXGRP;
    if (n->isdir()) {
        dir_node *d   = n->as_dir();
        p->st_nlink   = d->items().size();
        p->st_mode    = S_IFDIR | exe_perm | read_perm;
        p->st_size    = 0;
        p->st_blksize = 0;
        p->st_blocks  = 0;

        LOG().fmt("%s(%s, %p) -> dir", __func__, path, p);
    } else {
        file_node *f = n->as_file();
        p->st_nlink  = 1;
        p->st_mode   = S_IFREG | read_perm;
        auto r       = f->open();
        p->st_size   = r->remain();

        // used by $ du
        const int64_t blksize = 1 << 12;
        p->st_blksize         = blksize;
        p->st_blocks          = ceil_div(r->remain(), blksize);

        LOG().fmt("%s(%s, %p) -> file", __func__, path, p);
    }
    return 0;
}

int fuse::readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                  off_t off, fuse_file_info *fi)
{
    std::string info = show_fuse_file_info(fi);
    LOG().fmt("%s(%s, buf=%p, filler=%p, off=%d, fi=<%s>@%p)", __func__, path,
              buf, filler, off, info.c_str(), fi);
    dir_node *n = root.open(path)->as_dir();
    int cnt     = 0;
    for (auto &item : n->items()) {
        // LOG().fmt("%s += %s", __func__, item.c_str());
        if (filler(buf, item.c_str(), nullptr, 0) != 0) { return -ENOMEM; }
        ++cnt;
    }
    LOG().fmt("%s(%s, buf=%p, filler=%p, off=%d, fi=<%s>@%p) -> %d items",
              __func__, path, buf, filler, off, info.c_str(), fi, cnt);
    return 0;
}

int fuse::open(char const *path, fuse_file_info *fi)
{
    std::string info = show_fuse_file_info(fi);
    LOG().fmt("%s(%s, fi=<%s>@%p)", __func__, path, info.c_str(), fi);
    fi->fh = pool.get();
    LOG().fmt("%s(%s, fi=<%s>@%p) -> %d", __func__, path, info.c_str(), fi,
              fi->fh);
    return 0;
}

int fuse::read(const char *path, char *buf, size_t size, long off,
               fuse_file_info *fi)
{
    std::string info = show_fuse_file_info(fi);
    LOG().fmt("%s(%s, buf=%p, size=%zu, off=%zu, fi=<%s>@%p)", __func__, path,
              buf, size, off, info.c_str(), fi);
    node *n = root.open(path);
    if (n == nullptr || n->isdir()) { return -1; }
    file_node *f = n->as_file();
    auto r       = f->openat(off);
    size_t got   = r->read(buf, size);
    LOG().fmt("%s -> %zu", __func__, got);
    return got;
}

void fuse::run(const std::string &mount_path,
               const std::function<void(tree &)> &init_fs,
               const std::function<void()> &before_fuse_main, bool test)
{
    init_fs(fuse::root);
    // fuse::root.dump();

    fuse_operations ops;
    memset(&ops, 0, sizeof(ops));
    DEF_FUSE_FUNC(ops, getattr, fuse::getattr);
    DEF_FUSE_FUNC(ops, readdir, fuse::readdir);
    DEF_FUSE_FUNC(ops, open, fuse::open);
    DEF_FUSE_FUNC(ops, read, fuse::read);

    int uid = getuid();
    int gid = getgid();
    // must call before fuse_main
    LOG().fmt("%s, uid=%d, gid=%d", "starting mlfs ...", uid, gid);
    if (test) { return; }

    std::vector<std::string> args = {"tfrecord", mount_path};
    std::vector<char *> argv;
    for (auto &a : args) { argv.push_back(a.data()); }
    before_fuse_main();
    fuse_main(argv.size(), argv.data(), &ops, nullptr);
}
}  // namespace stdml::vfs
