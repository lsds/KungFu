#include <kungfu.h>

#include <sched.h>

namespace kungfu
{
int set_affinity(const Peer &peer)
{
    cpu_set_t cpu;
    const int nproc      = CPU_COUNT(&cpu);
    const int local_size = peer.LocalSize();
    if (local_size > nproc) {
        // can't set affinity
        return 0;
    }
    CPU_ZERO(&cpu);
    const int cores      = nproc / local_size;
    const int local_rank = peer.LocalRank();
    for (int i = 0; i < cores; ++i) { CPU_SET(local_rank * cores + i, &cpu); }
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpu);
}
}  // namespace kungfu
