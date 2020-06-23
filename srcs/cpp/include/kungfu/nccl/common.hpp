#pragma once
#include <map>
#include <string>

#include <kungfu.h>

enum KungFu_NCCLScope {
    KungFu_NCCL_GLOBAL,
    KungFu_NCCL_LOCAL,
};

namespace kungfu
{
// Workspace holds metadata for AllReduce, Reduce, Broadcast
struct Workspace {
    const void *sendbuf;
    void *recvbuf;
    const int count;
    const KungFu_Datatype dtype;
};

extern const std::map<std::string, KungFu_NCCLScope> _nccl_scopes;
}  // namespace kungfu
