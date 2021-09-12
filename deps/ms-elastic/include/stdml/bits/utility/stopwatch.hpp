#pragma once
#include <chrono>

namespace stdml
{
class stopwatch
{
    using C = std::chrono::high_resolution_clock;
    using T = std::chrono::time_point<C>;
    using D = std::chrono::duration<double>;

    T t0;

  public:
    stopwatch() : t0(C::now())
    {
    }

    D tick()
    {
        T t1 = C::now();
        D d = t1 - t0;
        t0 = t1;
        return d;
    }
};
}  // namespace stdml
