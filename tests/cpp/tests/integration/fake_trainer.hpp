#pragma once
#include <string>
#include <vector>

#include "testing.hpp"

std::string fake_grad_name(int i)
{
    return "NegotiatedGrad_" + std::to_string(i) + "/AllReduce";
}

template <typename T> struct fake_grad_t {
    std::string name;
    int count;
    // KungFu_Datatype dtype;
    std::vector<T> send_buf;
    std::vector<T> recv_buf;

    fake_grad_t(const std::string &name, int count)
        : name(name), count(count),
          //  dtype(kungfu::type_encoder::value<T>()),
          send_buf(count), recv_buf(count)
    {
    }
};

template <typename T> using grad_list_t = std::vector<fake_grad_t<T>>;

template <typename T>
grad_list_t<T> gen_fake_grads(const std::vector<int> &sizes)
{
    TRACE_SCOPE(__func__);
    int idx = 0;
    grad_list_t<T> grads;
    for (const auto &size : sizes) {
        fake_grad_t<T> g(fake_grad_name(idx++), size);
        grads.push_back(g);
    }
    return grads;
}

template <typename T>
grad_list_t<T> gen_fused_fake_grads(const std::vector<int> &sizes)
{
    TRACE_SCOPE(__func__);
    int total_size = 0;
    int idx        = 0;
    std::string fused_name;
    for (const auto &size : sizes) {
        fused_name += fake_grad_name(idx++);
        total_size += size;
    }
    grad_list_t<T> grads;
    fake_grad_t<T> g(fused_name, total_size);
    grads.push_back(g);
    return grads;
}
