#pragma once
#include <cstdio>
#include <mutex>

namespace stdml::vfs
{
class logger
{
    int no;
    FILE *lf;
    int t0;
    int pid;
    int ppid;
    int uid;
    int gid;

    std::mutex mu;

    void write_log(const char *msg);

  public:
    static logger &get(const char *filename);

    logger(const char *filename);

    template <typename... Args>
    void fmt(const char *format, const Args &... args)
    {
        char line[1 << 10];
        sprintf(line, format, args...);
        write_log(line);
    }
};
}  // namespace stdml::vfs
