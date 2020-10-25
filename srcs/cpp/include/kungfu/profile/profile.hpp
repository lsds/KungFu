#pragma once
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace kungfu
{
namespace profile
{
using clock_t    = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double>;
using instant_t  = std::chrono::time_point<clock_t>;

instant_t now() { return clock_t::now(); }

duration_t since(const instant_t &t0) { return now() - t0; }
}  // namespace profile

class ScheduledNcclAllReduceProfiler
{
    const std::string name_;

    std::vector<profile::duration_t> wait_durations_;
    std::vector<profile::duration_t> run_durations_;

  public:
    ScheduledNcclAllReduceProfiler(const std::string name)
        : name_(std::move(name))
    {
    }

    ~ScheduledNcclAllReduceProfiler()
    {
        char filename[256];
        std::sprintf(filename, "profile.%s.log", name_.c_str());
        std::replace(filename, filename + strlen(filename), '/', '_');
        FILE *fp = fopen(filename, "w");
        if (fp == nullptr) {
            fprintf(stderr, "failed to create file %s\n", filename);
            return;
        }
        const int n = wait_durations_.size();
        fprintf(fp, "%s\n", name_.c_str());
        fprintf(fp, "called %d times\n", n);
        fprintf(fp, "%s %s\n", "wait", "run");
        fprintf(fp, "%s\n", string('-', 80).c_str());
        for (int i = 0; i < n; ++i) {
            fprintf(fp, "%.3fms %.3fms\n",  //
                    wait_durations_.at(i).count() * 1000,
                    run_durations_.at(i).count() * 1000);
        }
        fclose(fp);
    }

    void wait_took(const profile::duration_t &d)
    {
        wait_durations_.push_back(d);
    }

    void run_took(const profile::duration_t &d) { run_durations_.push_back(d); }
};
}  // namespace kungfu

#define KUNGFU_MEASURE(stmt)                                                   \
    [&] {                                                                      \
        const auto t0 = kungfu::profile::now();                                \
        stmt;                                                                  \
        const auto d = kungfu::profile::since(t0);                             \
        return d;                                                              \
    }();
