#include <stdml/bits/vfs/logging.hpp>
#include <stdml/bits/vfs/node.hpp>

#define LOG() logger::get("node.log")

namespace stdml::vfs
{
simple_node::simple_node(std::string name, std::string text)
    : isdir_(false), text_(std::move(text))
{
    LOG().fmt("new file node of len=%d, name=%s", (int)text_.size(),
              name.c_str());
}

simple_node::simple_node(std::vector<std::string> item)
    : isdir_(true), items_(std::move(item))
{
    LOG().fmt("new dir node of %d items", (int)items_.size());
}

void simple_node::add(std::string item)
{
    items_.emplace_back(std::move(item));
}

const std::vector<std::string> &simple_node::items() const
{
    return items_;
}

bool simple_node::isdir() const
{
    return isdir_;
}

file_node *simple_node::as_file()
{
    return this;
}

dir_node *simple_node::as_dir()
{
    return this;
};
}  // namespace stdml::vfs
