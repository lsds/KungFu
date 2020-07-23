#include <iostream>

#include <kungfu/ncclv2/helper.hpp>

namespace kungfu
{
NCCLHelper_V2::NCCLHelper_V2(Peer *peer) : peer_(peer)
{
    std::cerr << __func__ << std::endl;
    init(KungFu_NCCL_GLOBAL);
    // init(KungFu_NCCL_LOCAL);
}

NCCLHelper_V2::~NCCLHelper_V2()
{
    std::cerr << __func__ << std::endl;  //
}

void NCCLHelper_V2::init(KungFu_NCCLScope scope)
{
    std::cerr << __func__ << ": " << scope << std::endl;
    auto *controller = NCCLController_V2::Create(peer_, scope);
    controllers_[scope].reset(controller);
    schedulers_[scope].reset(new NCCLScheduler_V2(controller));
}

void NCCLHelper_V2::SimpleAllReduce(Workspace w, std::function<void()> ready,
                                    std::string name,
                                    std::function<void()> done)
{
    TODO(__func__);
    ready();
    done();
}

void NCCLHelper_V2::BeginScheduleAllReduce(
    const std::vector<std::string> &names)
{
    auto *scheduler = schedulers_[KungFu_NCCL_GLOBAL].get();
    scheduler->BeginStep(names);
}

void NCCLHelper_V2::ScheduleAllReduce(Workspace w, std::function<void()> ready,
                                      std::string name,
                                      std::function<void()> done)
{
    // TODO(__func__ + (": " + name));
    auto *scheduler  = schedulers_.at(KungFu_NCCL_GLOBAL).get();
    auto *controller = controllers_.at(KungFu_NCCL_GLOBAL).get();
    scheduler->Enqueue(name, [=] {
        ready();
        controller->AllReduce(w);
        done();
    });
}

void NCCLHelper_V2::BeginScheduleHierarchicalAllReduce(
    const std::vector<std::string> &names)
{
    TODO(__func__);
}

void NCCLHelper_V2::ScheduleHierarchicalAllReduce(Workspace w,
                                                  std::function<void()> ready,
                                                  std::string reduce_op_name,
                                                  std::string bcast_op_name,
                                                  std::function<void()> done)
{
    TODO(__func__);
    ready();
    done();
}

// Peer _default_peer;
// NCCLHelper _default_nccl_helper(&_default_peer);
}  // namespace kungfu
