#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/common.hpp>
#include <kungfu/ncclv2/controller.hpp>
#include <kungfu/ncclv2/scheduler.hpp>

namespace kungfu
{
// provides interfaces for Tensorflow or PyTorch operators to access NCCL APIs
class NCCLHelper_V2
{
    Peer *peer_;

    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLScheduler_V2>> schedulers_;
    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLController_V2>> controllers_;

    void init(KungFu_NCCLScope scope);

  public:
    NCCLHelper_V2(Peer *peer_);
    ~NCCLHelper_V2();

    // should be called after resize cluster
    void Reset();

    void SimpleAllReduce(Workspace w, std::function<void()> ready,
                         std::string name, std::function<void()> done);

    void BeginScheduleAllReduce(const std::vector<std::string> &names);

    void ScheduleAllReduce(Workspace w, std::function<void()> ready,
                           std::string name, std::function<void()> done);

    void
    BeginScheduleHierarchicalAllReduce(const std::vector<std::string> &names);

    void ScheduleHierarchicalAllReduce(Workspace w, std::function<void()> ready,
                                       std::string reduce_op_name,
                                       std::string bcast_op_name,
                                       std::function<void()> done);
};

// extern Peer _default_peer;
}  // namespace kungfu

extern std::unique_ptr<kungfu::NCCLHelper_V2> _default_nccl_helper_v2;
