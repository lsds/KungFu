#pragma once
#include <cstdio>
#include <cstdlib>

template <typename E, E succ, typename Show> class error_checker
{
    const char *file_;
    const int line_;
    const std::string hint_;

  public:
    error_checker(const char *file, const int line) : file_(file), line_(line)
    {
    }

    error_checker(const char *file, const int line, const std::string hint)
        : file_(file), line_(line), hint_(std::move(hint))
    {
    }

    const error_checker &operator<<(const E &err) const
    {
        if (err != succ) {
            Show show_error;
            if (hint_.empty()) {
                std::fprintf(stderr, "%s::%d: %s\n", file_, line_,
                             show_error(err).c_str());
            } else {
                std::fprintf(stderr, "%s::%d: %s in %s\n", file_, line_,
                             show_error(err).c_str(), hint_.c_str());
            }
            std::exit(1);
        }
        return *this;
    }
};

#define KUNGFU_CHECK(checker) checker(__FILE__, __LINE__)

#define KUNGFU_CHECK_HINT(checker, ...) checker(__FILE__, __LINE__, __VA_ARGS__)
