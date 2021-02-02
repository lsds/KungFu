#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <kungfu.h>
#include <kungfu/numa/placement.hpp>

#include <sched.h>

namespace kungfu
{
extern std::vector<int> get_pu_numa_order();
extern size_t get_numa_node_count();

static std::string show(const std::vector<int> &arr)
{
    std::string s;
    for (auto x : arr) {
        if (!s.empty()) { s += ","; }
        s += std::to_string(x);
    }
    return s;
}

static int set_affinity(const std::vector<int> &cpu_order,
                        const size_t numa_node_count, const size_t local_rank,
                        const size_t local_size)
{
    const auto selected_cpus =
        select_cpus(cpu_order, numa_node_count, local_rank, local_size);
    fprintf(stderr, "binding local rank %d to %d cpus: %s\n",
            static_cast<int>(local_rank),
            static_cast<int>(selected_cpus.size()),
            show(selected_cpus).c_str());
    cpu_set_t cpu;
    CPU_ZERO(&cpu);
    for (auto i : selected_cpus) { CPU_SET(i, &cpu); }
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpu);
}

int set_affinity(const Peer &peer)
{
    std::vector<int> cpu_order;
    size_t numa_node_count = 1;
#ifdef KUNGFU_ENABLE_HWLOC
    cpu_order       = get_pu_numa_order();
    numa_node_count = get_numa_node_count();
    if (numa_node_count <= 0) {
        fprintf(stderr, "numa_node_count = %d, reset to 1\n",
                static_cast<int>(numa_node_count));
        numa_node_count = 1;
    }
    fprintf(stderr, "using numa cpu order: %s with %d numa nodes\n",
            show(cpu_order).c_str(), static_cast<int>(numa_node_count));
#else
    cpu_order.resize(std::thread::hardware_concurrency());
    std::iota(cpu_order.begin(), cpu_order.end(), 0);
#endif
    return set_affinity(cpu_order, numa_node_count, peer.LocalRank(),
                        peer.LocalSize());
}
}  // namespace kungfu
