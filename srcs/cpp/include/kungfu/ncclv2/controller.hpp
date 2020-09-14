#pragma once
#include <kungfu.h>
#include <kungfu/nccl/common.hpp>

namespace kungfu
{
//
class NCCLController_V2
{
  public:
    static NCCLController_V2 *Create(Peer *peer, KungFu_NCCLScope scope);

    virtual void Init()                 = 0;
    virtual void InitOnce()             = 0;
    virtual void Reduce(Workspace w)    = 0;
    virtual void Broadcast(Workspace w) = 0;
    virtual void AllReduce(Workspace w) = 0;
    virtual ~NCCLController_V2()        = default;
};
}  // namespace kungfu
