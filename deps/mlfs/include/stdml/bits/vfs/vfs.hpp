#pragma once
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <stdml/bits/vfs/node.hpp>

namespace stdml::vfs
{
class dict
{
    using K = std::string;
    std::unordered_map<K, int> ids;
    std::vector<K> words;

  public:
    int operator[](const K &k);

    K idx(int i) const;
};

class fd_pool
{
    std::map<int, int> fds_;

  public:
    int get();

    void put(int fd);
};

class path : public std::vector<int>
{
    using super = std::vector<int>;

  public:
    explicit path(std::vector<int> parts);

    bool isroot() const;

    path parent() const;

    int base() const;
};

class tree
{
    dict names_;

    std::map<path, std::unique_ptr<node>> nodes;

    node *getParent(path p);

    node *adddir(path p);

    node *addfile(path p, std::string text = "");

    std::string decode(const path &) const;

    path parse(const std::string &);

  public:
    tree();

    bool exists(const std::string &);

    node *mkdir(const std::string &);

    // touch a static file
    node *touch(const std::string &, const std::string &text = "");

    // touch a file with customizer handler
    void touch(const std::string &, node *);

    node *open(const std::string &);

    void dump() const;
};

class fileutil
{
  public:
    static void mkprefix(const std::string &, tree &);
};

void gen_example_fs(tree &r);
}  // namespace stdml::vfs
