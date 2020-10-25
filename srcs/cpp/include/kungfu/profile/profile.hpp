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
    std::vector<size_t> data_sizes_;

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
        fprintf(fp, "%s %s %s %s\n",  //
                "wait(ms)", "run(ms)", "size(KiB)", "rate(Gi/s)");
        fprintf(fp, "%s\n", std::string(80, '-').c_str());
        for (int i = 0; i < n; ++i) {
            const auto d1      = wait_durations_.at(i);
            const auto d2      = run_durations_.at(i);
            const auto size    = data_sizes_.at(i);
            constexpr float Ki = static_cast<float>(1 << 10);
            constexpr float Gi = static_cast<float>(1 << 30);
            const float rate   = (size / Gi) / d2.count();
            fprintf(fp, "%.3f %.3f %.3f, %.3f\n", d1.count() * 1000,
                    d2.count() * 1000, size / Ki, rate);
        }
        fclose(fp);
    }

    void wait_took(const profile::duration_t &d)
    {
        wait_durations_.push_back(d);
    }

    void run_took(const profile::duration_t &d, size_t size)
    {
        run_durations_.push_back(d);
        data_sizes_.push_back(size);
    }
};
}  // namespace kungfu

#define KUNGFU_MEASURE(stmt)                                                   \
    [&] {                                                                      \
        const auto t0 = kungfu::profile::now();                                \
        stmt;                                                                  \
        const auto d = kungfu::profile::since(t0);                             \
        return d;                                                              \
    }();
