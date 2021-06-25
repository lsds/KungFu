#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>

#include <kungfu.h>
#include <kungfu/python/c_api.h>
#include <kungfu/utils/trace.hpp>

DEFINE_TRACE_CONTEXT(kungfu);

std::unique_ptr<kungfu::Peer> _default_peer;

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
    _default_peer.reset(new kungfu::Peer);
    _init_affinity();
}

void kungfu_python_init_single_machine(int rank, int size)
{
    _default_peer.reset(new kungfu::Peer(rank, size));
    _init_affinity();
}

extern void kungfu_python_init_from_json(const char *pJson)
{
    _default_peer.reset(new kungfu::Peer(pJson));
    _init_affinity();
}

void kungfu_python_finialize() { _default_peer.reset(nullptr); }

uint64_t kungfu_uid() { return _default_peer->Uid(); }

int kungfu_detached() { return _default_peer->Detached(); }

int kungfu_rank() { return _default_peer->Rank(); }

int kungfu_size() { return _default_peer->Size(); }

int kungfu_local_rank() { return _default_peer->LocalRank(); }

int kungfu_local_size() { return _default_peer->LocalSize(); }

void kungfu_barrier() { _default_peer->Barrier(); }

int kungfu_propose_new_size(int new_size)
{
    return _default_peer->ProposeNewSize(new_size);
}

#ifdef KUNGFU_ENABLE_AFFINITY
void kungfu_set_affinity() { kungfu::set_affinity(*_default_peer); }
#endif

int kungfu_check_interference() { return _default_peer->CheckInterference(); }

void kungfu_calc_stats() { return _default_peer->CalcStats(); }

void kungfu_log_stats() { return _default_peer->LogStats(); }

void kungfu_print_strategy_stats() { _default_peer->PrintStategyStats(); }
