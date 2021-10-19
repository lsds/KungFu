#include <algorithm>
#include <experimental/iterator>
#include <iostream>
#include <sstream>

#include <stdml/bits/vfs/logging.hpp>
#include <stdml/bits/vfs/vfs.hpp>

#define LOG() logger::get("tree.log")

namespace stdml::vfs
{
tree::tree() { mkdir("/"); }

path tree::parse(const std::string &s)
{
    std::vector<int> parts;
    std::stringstream ss(s);
    for (std::string line; std::getline(ss, line, '/');) {
        if (!line.empty()) {
            int x = names_[line];
            parts.push_back(x);
        }
    }
    return path(std::move(parts));
}

std::string tree::decode(const path &p) const
{
    std::stringstream ss;
    ss << '/';
    std::transform(p.begin(), p.end(),
                   std::experimental::make_ostream_joiner(ss, '/'),
                   [&](int i) { return names_.idx(i); });
    return ss.str();
}

node *tree::getParent(path p)
{
    path pp = p.parent();
    if (nodes.count(pp) == 0) {
        std::cerr << "not exist: " << decode(pp) << std::endl;
    }
    node *up = nodes.at(pp).get();
    return up;
}

node *tree::adddir(path p)
{
    std::vector<std::string> empty;
    node *n = new simple_node(empty);
    nodes[p].reset(n);
    LOG().fmt("%s(%s)", __func__, decode(p).c_str());
    return n;
}

node *tree::addfile(path p, std::string text)
{
    node *n = new simple_node(names_.idx(p.base()), std::move(text));
    nodes[p].reset(n);
    return n;
}

bool tree::exists(const std::string &s)
{
    path p = parse(s);
    return nodes.count(p) > 0;
}

node *tree::mkdir(const std::string &s)
{
    path p = parse(s);
    if (nodes.count(p) > 0) { throw std::runtime_error(s + " exists"); }
    if (p.isroot()) { return adddir(p); }
    node *up = getParent(p);
    node *n  = adddir(p);
    up->as_dir()->add(names_.idx(p.base()));
    return n;
}

node *tree::touch(const std::string &s, const std::string &text)
{
    path p = parse(s);
    if (nodes.count(p) > 0) { throw std::runtime_error(s + " exists"); }
    node *up = getParent(p);
    node *n  = addfile(p, text);
    up->as_dir()->add(names_.idx(p.base()));
    return n;
}

void tree::touch(const std::string &s, node *n)
{
    path p = parse(s);
    if (nodes.count(p) > 0) { throw std::runtime_error(s + " exists"); }
    node *up = getParent(p);
    nodes[p].reset(n);
    up->as_dir()->add(names_.idx(p.base()));
}

node *tree::open(const std::string &s)
{
    path p = parse(s);
    if (nodes.count(p) == 0) { return nullptr; }
    return nodes.at(p).get();
}

void tree::dump() const
{
    for (const auto &[p, n] : nodes) {
        const std::string k = decode(p);
        printf("%p : %s\n", n.get(), k.c_str());
        if (n->isdir()) {
            for (const auto &i : n->as_dir()->items()) {
                printf("                   - %s\n", i.c_str());
            }
        }
    }
    printf("%d nodes\n", (int)nodes.size());
    fflush(stdout);
}
}  // namespace stdml::vfs
