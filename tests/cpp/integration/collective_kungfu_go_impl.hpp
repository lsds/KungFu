#pragma once
#include <functional>

#include "testing.hpp"

class kungfu_go_collective
{
    kungfu_world _kungfu_world;

    const int _rank;
    const int _cluster_size;

  public:
    kungfu_go_collective(int argc, char *argv[]) : kungfu_go_collective() {}

    kungfu_go_collective()
        : _rank(getSelfRank()), _cluster_size(getTestClusterSize())
    {
    }

    ~kungfu_go_collective() {}

    bool is_root() const { return _rank == 0; }

    int rank() const { return _rank; }

    int cluster_size() const { return _cluster_size; }

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
