#pragma once
#include <string>
#include <thread>
#include <vector>

#include "collective.hpp"
#include "testing.hpp"

std::string fake_grad_name(int i)
{
    return "NegotiatedGrad_" + std::to_string(i) + "/AllReduce";
}

template <typename T> struct fake_buffer_t {
    using value_type = T;
    std::string name;
    int count;
    std::vector<T> send_buf;
    std::vector<T> recv_buf;

    fake_buffer_t(const std::string &name, int count)
        : name(name), count(count), send_buf(count), recv_buf(count)
    {
    }
};

template <typename T> using grad_list_t = std::vector<fake_buffer_t<T>>;

template <typename T>
grad_list_t<T> gen_fake_grads(const std::vector<int> &sizes)
{
    grad_list_t<T> grads;
    int idx = 0;
    for (const auto &size : sizes) {
        fake_buffer_t<T> g(fake_grad_name(idx++), size);
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
    fake_buffer_t<T> g(fused_name, total_size);
    grads.push_back(g);
    return grads;
}

template <bool async = true> class fake_minibatch_runner_t
{
    using ms = std::chrono::milliseconds;

    const bool _fake_train = false;
    const ms _fake_sleep_duration;

    template <typename T, typename Collective>
    void run_sync(grad_list_t<T> &grads, Collective &comm) const
    {
        for (auto &g : grads) {
            TRACE_SCOPE("run_sync::collective_all_reduce");
            collective_all_reduce(g.send_buf.data(), g.recv_buf.data(), g.count,
                                  g.name.c_str(), comm);
        }
    }

    template <typename T, typename Collective>
    void run_async(grad_list_t<T> &grads, Collective &comm) const
    {
        std::vector<Waiter> waiters(grads.size());
        int idx = 0;
        for (auto &g : grads) {
            collective_all_reduce(g.send_buf.data(), g.recv_buf.data(), g.count,
                                  g.name.c_str(), comm,
                                  [i = idx++, &waiters] { waiters[i].done(); });
        }
        for (auto &w : waiters) { w.wait(); }
    }

  public:
    fake_minibatch_runner_t(int batch_size, double image_per_sec)
        : _fake_sleep_duration((int)(batch_size * 1e3 / image_per_sec))
    {
        if (_fake_train) {
            const int d = _fake_sleep_duration.count();
            fprintf(stderr, "fake_sleep_duration: %dms\n", d);
        }
    }

    template <typename T, typename Collective>
    void operator()(grad_list_t<T> &grads, Collective &comm) const
    {
        if (_fake_train) {
            TRACE_SCOPE("train stage");
            std::this_thread::sleep_for(_fake_sleep_duration);
        }
        {
            TRACE_SCOPE("AllReduce stage");
            if (async) {
                run_async(grads, comm);
            } else {
                run_sync(grads, comm);
            }
        }
    }
};

class fake_trainer_t
{
    const bool is_root;
    const int cluster_size;

    const int n_iters;
    const int step_per_iter;
    const int batch_size;

  public:
    fake_trainer_t(bool is_root, int cluster_size, int n_iters,
                   int step_per_iter, int batch_size)
        : is_root(is_root), cluster_size(cluster_size), n_iters(n_iters),
          step_per_iter(step_per_iter), batch_size(batch_size)
    {
    }

    template <typename T, typename Step, typename Collective>
    void operator()(const Step &minibatch, grad_list_t<T> &grads,
                    Collective &comm)
    {
        const auto t0 = testing::now();
        int step      = 0;
        for (int i = 0; i < n_iters; ++i) {
            for (int j = 0; j < step_per_iter; ++j) {
                ++step;
                TRACE_SCOPE("mini batch");
                minibatch(grads, comm);
            }
            if (is_root) { fprintf(stderr, "after %d steps\n", step); }
        }
        const auto d = testing::since(t0);
        const double img_per_sec =
            n_iters * step_per_iter * batch_size / d.count();
        fprintf(stderr, "Img/sec %.2f per worker, np=%d\n", img_per_sec,
                cluster_size);
    }
};

template <typename Collective>
void run_experiment(bool is_root, int cluster_size,
                    const std::vector<int> &grad_sizes, Collective &comm)
{
    const int batch_size       = 32;
    const double image_per_sec = 185;
    const int n_iters          = 11;
    const int step_per_iter    = 10;

    constexpr bool async = false;
    fake_minibatch_runner_t<async> minibatch(batch_size, image_per_sec);
    fake_trainer_t train(is_root, cluster_size, n_iters, step_per_iter,
                         batch_size);

    bool fuse_grads = true;
    using T         = float;
    auto grads      = fuse_grads ? gen_fused_fake_grads<T>(grad_sizes)
                            : gen_fake_grads<T>(grad_sizes);
    train(minibatch, grads, comm);
}
