#include <cinttypes>

#include <stdml/bits/utility/stat.hpp>

namespace stdml
{
const char *show_size(int64_t n)
{
    constexpr int64_t Ki = 1 << 10;
    constexpr int64_t Mi = 1 << 20;
    constexpr int64_t Gi = 1 << 30;
    constexpr int64_t Ti = (int64_t)1 << 40;
    static char line[32];
    if (n < Ki) {
        sprintf(line, "%d B", (int)n);
    } else if (n < Mi) {
        sprintf(line, "%.3f KiB", (double)n / Ki);
    } else if (n < Gi) {
        sprintf(line, "%.3f MiB", (double)n / Mi);
    } else if (n < Ti) {
        sprintf(line, "%.3f GiB", (double)n / Gi);
    } else {
        sprintf(line, "%.3f TiB", (double)n / Ti);
    }
    return line;
}
}  // namespace stdml
