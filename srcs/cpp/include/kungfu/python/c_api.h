#pragma once
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

extern void kungfu_python_init();
extern void kungfu_python_init_single_machine(int rank, int size);
extern void kungfu_python_init_from_json(const char *pJson);
extern void kungfu_python_init_nccl();

extern void kungfu_python_finialize();
extern void kungfu_python_finialize_nccl();

extern int kungfu_get_cuda_index();

// helpers APIs to access kungfu without tensorflow operators
extern uint64_t kungfu_uid();
extern int kungfu_detached();
extern int kungfu_rank();        // get current rank
extern int kungfu_size();        // get current size
extern int kungfu_local_rank();  // get current local rank
extern int kungfu_local_size();  // get current local size
extern void kungfu_barrier();
extern void kungfu_set_affinity();

extern int kungfu_propose_new_size(int new_size);

extern int kungfu_check_interference();

extern void kungfu_calc_stats();

extern void kungfu_log_stats();

extern void kungfu_print_strategy_stats();

// unstable APIs
extern void kungfu_resize(int n, char *changed, char *detached);
extern void kungfu_resize_from_url(char *p_changed, char *p_detached);

#ifdef __cplusplus
}
#endif

#include <kungfu/python/collective.h>

namespace kungfu
{
class Peer;
class NCCLHelper;
}  // namespace kungfu

extern std::unique_ptr<kungfu::Peer> _default_peer;
extern std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;
