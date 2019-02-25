#pragma once
#include <cstdio>
#include <cstdlib>

template <typename E, E succ, typename Show> struct error_checker {
    const char *file;
    const int line;

    error_checker(const char *file, const int line) : file(file), line(line) {}

    const error_checker &operator<<(const E &err) const
    {
        if (err != succ) {
            Show show_error;
            fprintf(stderr, "%s::%d: %s\n", file, line,
                    show_error(err).c_str());
            exit(1);
        }
        return *this;
    }
};

#define CHECK(checker) checker(__FILE__, __LINE__)
