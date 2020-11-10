#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <kungfu.h>

#include <sched.h>

namespace kungfu
{
extern std::vector<int> get_pu_numa_order();

std::string show(const std::vector<int> &arr)
{
    std::string s;
    for (auto x : arr) {
        if (!s.empty()) { s += ","; }
        s += std::to_string(x);
    }
    return s;
}

int set_affinity(const std::vector<int> &cpu_order, const size_t local_rank,
                 const size_t local_size)
{
    if (cpu_order.size() < local_size) {
        fprintf(stderr, "no enough cpus to bind\n");
        return 0;
    }
    const int cores = cpu_order.size() / local_size;
    std::vector<int> selected_cpus(cores);
    std::copy(cpu_order.begin() + local_rank * cores,
              cpu_order.begin() + (local_rank + 1) * cores,
              selected_cpus.begin());
    fprintf(stderr, "binding local rank %d to %d cpus: %s\n",
            static_cast<int>(local_rank),
            static_cast<int>(selected_cpus.size()),
            show(selected_cpus).c_str());
    cpu_set_t cpu;
    CPU_ZERO(&cpu);
    for (int i = 0; i < cores; ++i) { CPU_SET(selected_cpus[i], &cpu); }
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpu);
}

int set_affinity(const Peer &peer)
{
    std::vector<int> cpu_order;
#ifdef KUNGFU_ENABLE_HWLOC
    cpu_order = get_pu_numa_order();
    fprintf(stderr, "using numa cpu order: %s\n", show(cpu_order).c_str());
#else
    cpu_order.resize(std::thread::hardware_concurrency());
    std::iota(cpu_order.begin(), cpu_order.end(), 0);
#endif
    return set_affinity(cpu_order, peer.LocalRank(), peer.LocalSize());
}
}  // namespace kungfu
