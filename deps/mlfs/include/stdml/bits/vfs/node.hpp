#pragma once
#include <memory>
#include <string>
#include <vector>

namespace stdml::vfs
{
class simple_node;

class file_reader
{
  public:
    virtual ~file_reader() = default;

    virtual size_t remain() const = 0;

    virtual size_t read(char *buf, size_t limit) = 0;
};

class ptr_reader : public file_reader
{
    const char *ptr;
    const size_t len;

    size_t pos;

  public:
    ptr_reader(const char *p, size_t size, size_t off = 0)
        : ptr(p), len(size), pos(off)
    {
    }

    size_t remain() const override { return len - pos; }

    size_t read(char *buf, size_t limit) override
    {
        size_t n        = std::min<size_t>(remain(), limit);
        const char *bgn = ptr + pos;
        std::copy(bgn, bgn + n, buf);
        pos += n;
        return n;
    }
};

class file_node
{
  public:
    virtual std::unique_ptr<file_reader> openat(size_t off) = 0;

    std::unique_ptr<file_reader> open() { return openat(0); }
};

class dir_node
{
  public:
    virtual const std::vector<std::string> &items() const = 0;
    virtual void add(std::string item)                    = 0;
};

class node
{
  public:
    virtual ~node() = default;

    virtual bool isdir() const = 0;

    virtual file_node *as_file() = 0;
    virtual dir_node *as_dir()   = 0;
};

class simple_node : public node, public file_node, public dir_node
{
    bool isdir_;
    std::string text_;
    std::vector<std::string> items_;

  public:
    simple_node(std::string name, std::string text);
    simple_node(std::vector<std::string> item);

    bool isdir() const override;
    file_node *as_file() override;
    dir_node *as_dir() override;

    void add(std::string item) override;
    const std::vector<std::string> &items() const override;

    std::unique_ptr<file_reader> openat(size_t off) override
    {
        if (isdir_) { throw std::runtime_error("not a file"); }
        ptr_reader *pr = new ptr_reader(text_.data(), text_.size(), off);
        return std::unique_ptr<file_reader>(pr);
    }
};
}  // namespace stdml::vfs
