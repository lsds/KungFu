#pragma once
#include <functional>

#include "testing.hpp"

struct kungfu_go_collective {
    kungfu_world _kungfu_world;

    kungfu_go_collective() {}

    ~kungfu_go_collective() {}

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
};
