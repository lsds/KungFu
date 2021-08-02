#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>

#include <kungfu.h>
#include <kungfu/python/c_api.h>
#include <kungfu/utils/trace.hpp>

DEFINE_TRACE_CONTEXT(kungfu);

bool parse_bool_env(const char *name)
{
    const char *ptr = std::getenv(name);
    if (ptr == nullptr) { return false; }
    std::string val(ptr);
    std::transform(val.begin(), val.end(), val.begin(), tolower);
    return val == "1" || val == "true" || val == "on";
}

static void _init_affinity()
{
#ifdef KUNGFU_ENABLE_AFFINITY
    if (parse_bool_env("KUNGFU_USE_AFFINITY")) { kungfu_set_affinity(); }
#endif
}

void kungfu_python_init()
{
    kungfu::Peer::GetDefault(true);
    _init_affinity();
}

void kungfu_python_init_single_machine(int rank, int size)
{
    auto &peer = kungfu::Peer::GetDefault();
    peer.reset(new kungfu::Peer(rank, size));
    _init_affinity();
}

void kungfu_python_finialize() { kungfu::Peer::GetDefault().reset(nullptr); }

uint64_t kungfu_uid() { return kungfu::Peer::GetDefault()->Uid(); }

int kungfu_detached() { return kungfu::Peer::GetDefault()->Detached(); }

int kungfu_rank() { return kungfu::Peer::GetDefault()->Rank(); }

int kungfu_size() { return kungfu::Peer::GetDefault()->Size(); }

int kungfu_local_rank() { return kungfu::Peer::GetDefault()->LocalRank(); }

int kungfu_local_size() { return kungfu::Peer::GetDefault()->LocalSize(); }

void kungfu_barrier() { kungfu::Peer::GetDefault()->Barrier(); }

int kungfu_propose_new_size(int new_size)
{
    return kungfu::Peer::GetDefault()->ProposeNewSize(new_size);
}

#ifdef KUNGFU_ENABLE_AFFINITY
void kungfu_set_affinity()
{
    kungfu::set_affinity(*kungfu::Peer::GetDefault());
}
#endif

int kungfu_check_interference()
{
    return kungfu::Peer::GetDefault()->CheckInterference();
}

void kungfu_calc_stats() { return kungfu::Peer::GetDefault()->CalcStats(); }

void kungfu_log_stats() { return kungfu::Peer::GetDefault()->LogStats(); }

void kungfu_print_strategy_stats()
{
    kungfu::Peer::GetDefault()->PrintStategyStats();
}
