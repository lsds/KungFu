#pragma once
#include <cstddef>
#include <vector>

namespace kungfu
{
std::vector<int> select_cpus(const std::vector<int> &all_cpus,
                             const size_t numa_node_count,
                             const size_t local_rank, const size_t local_size);
}
