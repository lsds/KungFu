#pragma once
#include <functional>

#include "testing.hpp"

class kungfu_go_collective
{
    kungfu_world _kungfu_world;

  public:
    kungfu_go_collective(int argc, char *argv[]) : kungfu_go_collective() {}

    kungfu_go_collective() {}

    ~kungfu_go_collective() {}

    bool is_root() const { return _kungfu_world.Rank() == 0; }

    int rank() const { return _kungfu_world.Rank(); }

    int cluster_size() const { return _kungfu_world.ClusterSize(); }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char *name)
    {
        _kungfu_world.AllReduce(send_buf, recv_buf, count,
                                kungfu::type_encoder::value<T>(), KungFu_SUM,
                                name);
    }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char *name, std::function<void()> done)
    {
        _kungfu_world.AllReduce(send_buf, recv_buf, count,
                                kungfu::type_encoder::value<T>(), KungFu_SUM,
                                name, done);
    }

    template <typename T> void bcast(T *buf, size_t count, const char *name)
    {
        _kungfu_world.Broadcast(buf, buf, count,
                                kungfu::type_encoder::value<T>(), name);
    }
};
