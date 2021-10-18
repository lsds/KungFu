#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <stdml/bits/vfs/logging.hpp>
#include <unistd.h>

namespace stdml::vfs
{
static int unixtime()
{
    std::time_t t = std::time(nullptr);
    return t;
}

logger &logger::get(const char *filename)
{
    static std::mutex mu;
    static std::unordered_map<std::string, std::unique_ptr<logger>> logs;
    {
        std::lock_guard<std::mutex> _lk(mu);
        if (logs.count(filename) == 0) {
            logs.emplace(filename, new logger(filename));
        }
        return *logs.at(filename);
    }
}

logger::logger(const char *filename)
    : no(0),
      lf(std::fopen(filename, "w")),
      t0(unixtime()),
      pid(getpid()),
      ppid(getppid()),
      uid(getuid()),
      gid(getgid())
{
    if (lf != nullptr) {
        std::setvbuf(lf, nullptr, _IONBF, 0);
    } else {
        // perror("fopen");
        // exit(1);
    }
}

void logger::write_log(const char *msg)
{
    int t = unixtime();
    int pid = getpid();
    int ppid = getppid();

    bool log_tid = true;
    bool log_pid = true;
    bool log_timestamp = true;

    std::stringstream ss;
    ss << "[i]";
    if (log_pid) {
        ss << ' ' << pid << '/' << ppid;
    }
    if (log_tid) {
        thread_local std::thread::id tid = std::this_thread::get_id();
        ss << ' ' << tid;
    }
    if (log_timestamp) {
        ss << ' ' << t;
    }

    ss << ' ' << msg;
    {
        std::lock_guard<std::mutex> _lk(mu);
        ++no;
        fprintf(lf, "%s\n", ss.str().c_str());
    }
}
}  // namespace stdml::vfs
