#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/common.hpp>
#include <kungfu/nccl/helper.hpp>

extern "C" {
extern void kungfu_python_init();
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

extern int kungfu_propose_new_size(int new_size);

typedef enum KungFu_NCCLScope KungFu_NCCLScope;
}

extern std::unique_ptr<kungfu::Peer> _default_peer;
extern std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;
