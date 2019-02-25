#pragma once
#include <functional>

template <typename T, typename Collective>
void collective_all_reduce(const T *send_buf, T *recv_buf, size_t count,
                           const char *name, Collective &comm)
{
    comm.all_reduce(send_buf, recv_buf, count, name);
}

template <typename T, typename Collective>
void collective_all_reduce(const T *send_buf, T *recv_buf, size_t count,
                           const char *name, Collective &comm,
                           std::function<void()> done)
{
    comm.all_reduce(send_buf, recv_buf, count, name, done);
}
