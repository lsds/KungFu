#include <iostream>

#include <kungfu/nccl/controller.hpp>
#include <kungfu/ncclv2/helper.hpp>

namespace kungfu
{
NCCLHelper_V2::NCCLHelper_V2(Peer *peer) : peer_(peer)
{
    DBG(__func__);
    init(KungFu_NCCL_GLOBAL);  // TODO: lazy init
    init(KungFu_NCCL_LOCAL);   // TODO: lazy init
}

NCCLHelper_V2::~NCCLHelper_V2() { DBG(__func__); }

void NCCLHelper_V2::init(KungFu_NCCLScope scope)
{
    DBG(__func__ + std::string(" : ") + std::to_string(scope));
    auto *controller = NCCLController_V2::Create(peer_, scope);
    controllers_[scope].reset(controller);
    schedulers_[scope].reset(new NCCLScheduler_V2(controller));
}

void NCCLHelper_V2::BeginScheduleAllReduce(
    const std::vector<std::string> &names)
{
    auto *scheduler = schedulers_.at(KungFu_NCCL_GLOBAL).get();
    scheduler->BeginStep(names);
}

void NCCLHelper_V2::ScheduleAllReduce(Workspace w, std::function<void()> ready,
                                      std::string name,
                                      std::function<void()> done)
{
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
    DBG(__func__ + std::string(" with ") + std::to_string(names.size()) +
        " names");
    auto *scheduler = schedulers_.at(KungFu_NCCL_LOCAL).get();
    scheduler->BeginStep(names);
}

void NCCLHelper_V2::ScheduleHierarchicalAllReduce(Workspace w,
                                                  std::function<void()> ready,
                                                  std::string reduce_op_name,
                                                  std::string allreduce_op_name,
                                                  std::string bcast_op_name,
                                                  std::function<void()> done)
{
    auto *scheduler  = schedulers_.at(KungFu_NCCL_LOCAL).get();
    auto *controller = controllers_.at(KungFu_NCCL_LOCAL).get();

    Workspace w_reduce = w;

    Workspace w_all_reduce = w;
    w_all_reduce.sendbuf   = w.recvbuf;  // inplace

    Workspace w_bcast = w;
    w_bcast.sendbuf   = w.recvbuf;  // inplace

    scheduler->Enqueue(reduce_op_name, [=] {
        ready();
        controller->Reduce(w_reduce);
        CrossAllReduceGpu(w_all_reduce, KungFu_SUM, allreduce_op_name, [=] {
            scheduler->Enqueue(bcast_op_name, [=] {
                controller->Broadcast(w_bcast);
                done();
            });
        });
    });
}
}  // namespace kungfu
