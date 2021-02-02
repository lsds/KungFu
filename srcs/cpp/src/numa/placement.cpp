#include <kungfu/numa/placement.hpp>

namespace kungfu
{
static int ceil_div(int a, int b) { return a / b + (a % b ? 1 : 0); }

std::vector<int> select_cpus(const std::vector<int> &all_cpus,
                             const size_t numa_node_count,
                             const size_t local_rank, const size_t local_size)
{
    const auto n     = ceil_div(all_cpus.size(), numa_node_count);
    const int i      = local_rank / ceil_div(local_size, numa_node_count);
    const auto begin = all_cpus.begin() + n * i;
    const auto end   = std::min(all_cpus.end(), begin + n);
    return std::vector<int>(begin, end);
}
}  // namespace kungfu
