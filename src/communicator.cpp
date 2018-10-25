#include "communicator.h"

#include <cstdlib>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "logger.h"
#include "message.h"

class Endpoint
{
  public:
    Endpoint(const std::string &addr) : addr_(addr) {}

    // std::ofstream &addr() { return fs; }

    std::string addr() { return addr_; }

  private:
    // std::ofstream fs;
    const std::string addr_;
};

template <typename T>
void accumulate_gradient(void *gradient, const void *delta, int n)
{
    T *g = reinterpret_cast<T *>(gradient);
    const T *d = reinterpret_cast<const T *>(delta);
    std::transform(g, g + n, d, g, std::plus<T>());
}

void accumulate_gradient(void *g, const void *d, size_t size, int data_type)
{
    switch (data_type) {
    // FIXME: assume data_type is float
    default:
        accumulate_gradient<float>(g, d, size / sizeof(float));
    }
}

class AgentImpl : public Agent
{
  public:
    AgentImpl() {}

    void add_peer(Endpoint ep)
    {
        std::lock_guard<std::mutex> _(mu_);
        peers_.push_back(std::move(ep));
    }

    // only from local
    void push(const std::string &name, const void *data, size_t n) override
    {
        std::lock_guard<std::mutex> _(mu_);
        LOG(INFO) << name << " pushed, size: " << n;
        ensure_buffers(name, n);

        std::memcpy(local_gradients_.at(name).data(), data,
                    n);  // update local, overriwrite current local
        accumulate_gradient(global_gradients_.at(name).data(), data, n,
                            0);  // add to global

        broadcast(name);  // TODO: make it async
    }

    // only from remote, called async
    void recv(const std::string &name, const void *data, size_t n) override
    {
        std::lock_guard<std::mutex> _(mu_);
        LOG(INFO) << name << " received, size: " << n;
        ensure_buffers(name, n);
        accumulate_gradient(global_gradients_.at(name).data(), data, n, 0);
    }

    void pull(const std::string &name, void *data, size_t n) override
    {
        std::lock_guard<std::mutex> _(mu_);
        LOG(INFO) << "pulling " << name << ", size: " << n;
        if (global_gradients_.count(name) <= 0) {
            LOG(WARNING) << name << " not pushed";
            return;
        }

        auto &buffer = global_gradients_.at(name);
        assert(buffer.size() == n);
        std::memcpy(data, buffer.data(), n);
        buffer.clear();
    }

  private:
    void broadcast(const std::string &name)
    {
        // std::lock_guard<std::mutex> _(mu_);

        const auto &buffer = local_gradients_.at(name);
        kungfu::PushRequest req;
        req.name = name;
        req.size = buffer.size();
        req.data = buffer.data();

        for (auto &p : peers_) {
            // TODO: use multi-cast
            LOG(INFO) << "send gradient to peer: " << p.addr();
            kungfu::sendTo(req, p);
        }
    }

    void ensure_buffers(const std::string &name, int n)
    {
        if (local_gradients_.count(name) <= 0) {
            local_gradients_.emplace(name, kungfu::TensorBuffer(n));
        }
        if (global_gradients_.count(name) <= 0) {
            global_gradients_.emplace(name, kungfu::TensorBuffer(n));
        }
    }

    std::unordered_map<std::string, kungfu::TensorBuffer>
        local_gradients_;  // gradients from local worker

    // FIXME: use mean, or use model
    std::unordered_map<std::string, kungfu::TensorBuffer>
        global_gradients_;  // sum of gradients from local worker and peers

    std::vector<Endpoint> peers_;
    std::mutex mu_;
};

AgentImpl local_agent;

/*! Agentd is a daemon */
class AgentdImpl
{
  public:
    AgentdImpl(AgentImpl &agent, const std::string &endpoint,
               const std::vector<std::string> &peers)
        : agent_(agent)
    {
        for (auto p : peers) { agent_.add_peer(Endpoint(p)); }
    }

    ~AgentdImpl() { stop(); }

    void start()
    {
        LOG(INFO) << "starting Agentd thread";
        _th.reset(new std::thread([]() {
            LOG(INFO) << "in Agentd thread";
            //
        }));
    }

    void stop() { _th->join(); }

  private:
    AgentImpl &agent_;
    std::unique_ptr<std::thread> _th;
};

std::string get_env(const char *name)
{
    auto ptr = std::getenv(name);
    if (ptr) { return std::string(ptr); }
    return "";
}

std::vector<std::string> split(const std::string &text, const char sep)
{
    std::vector<std::string> lines;
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line, sep)) { lines.push_back(line); }
    return lines;
}

AgentdImpl *init_agentd()
{
    constexpr const char *KUNGFU_PEERS = "KUNGFU_PEERS";
    constexpr const char *KUNGFU_TASK = "KUNGFU_TASK";

    const auto kf_all_peers = split(get_env(KUNGFU_PEERS), ',');
    const auto kf_task = get_env(KUNGFU_TASK);

    LOG(INFO) << "KUNGFU_PEERS: " << kf_all_peers.size();
    LOG(INFO) << "KUNGFU_TASK: " << kf_task;

    std::vector<std::string> kf_other_peers;
    for (auto &p : kf_all_peers) {
        if (p != kf_task) { kf_other_peers.push_back(p); }
    }

    AgentdImpl *agentd = new AgentdImpl(local_agent, kf_task, kf_other_peers);
    agentd->start();
    return agentd;
}

std::unique_ptr<AgentdImpl> _agentd(init_agentd());

Agent *Agent::get_instance() { return &local_agent; }
