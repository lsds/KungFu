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

#define KUNGFU_DEBUG 1

#ifdef KUNGFU_DEBUG
#include <iostream>
#include <mutex>
#include <thread>

namespace kungfu
{
static std::mutex dbg_mu;

inline void TODO(const std::string &name)
{
    std::lock_guard<std::mutex> lk(dbg_mu);
    std::cerr << "TODO: " << name << std::endl;
}

inline void LOG_THREAD(const std::string &name)
{
    std::lock_guard<std::mutex> lk(dbg_mu);
    const auto tid = std::this_thread::get_id();
    std::cerr << "called in thread " << tid << " :: " << name << std::endl;
}

inline void DBG(const std::string &msg)
{
    std::lock_guard<std::mutex> lk(dbg_mu);
    std::cerr << "[DBG] " << msg << std::endl;
}

}  // namespace kungfu
#endif
