#pragma once
#include <algorithm>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "collective.hpp"
#include "testing.hpp"

std::string fake_grad_name(int i)
{
    return "NegotiatedGrad_" + std::to_string(i) + "/AllReduce";
}

template <typename T> struct fake_cpu_buffer_t {
    using value_type = T;
    const std::string name;
    const int count;

    std::vector<T> send_buf;
    std::vector<T> recv_buf;
    int recv_count;

    const T *effective_data() const
    {
        if (recv_count == 0) {
            return send_buf.data();
        } else {
            return recv_buf.data();
        }
    }

    fake_cpu_buffer_t(const std::string &name, int count)
        : name(name), count(count), send_buf(count), recv_buf(count),
          recv_count(0)
    {
    }

    void reset(T x)
    {
        recv_count = 0;
        std::fill(send_buf.begin(), send_buf.end(), x);
        const T magic = static_cast<T>(-12345678);
        std::iota(recv_buf.begin(), recv_buf.end(), magic);
    }

    bool check(T origin, T result) const
    {
        for (auto x : send_buf) {
            if (x != origin) { return false; }
        }
        for (auto x : recv_buf) {
            if (x != result) { return false; }
        }
        return true;
    }

    void recv_self()
    {
        std::copy(send_buf.begin(), send_buf.end(), recv_buf.begin());
        ++recv_count;
    }

    void recv_into(const fake_cpu_buffer_t &sender)
    {
        const T *src = sender.effective_data();
        // Assuming count == sender.count
        std::copy(src, src + count, recv_buf.data());
        ++recv_count;
    }

    void recv_onto(const fake_cpu_buffer_t &sender)
    {
        const T *src = sender.effective_data();
        // Assuming count == sender.count
        std::transform(src, src + count, effective_data(), recv_buf.data(),
                       std::plus<T>());
        ++recv_count;
    }
};

template <typename buffer_t>
std::vector<buffer_t> gen_fake_grads(const std::vector<int> &sizes)
{
    TRACE_SCOPE(__func__);
    std::vector<buffer_t> grads;
    int idx = 0;
    for (const auto &size : sizes) {
        grads.push_back(buffer_t(fake_grad_name(idx++), size));
    }
    return grads;
}

template <typename buffer_t>
std::vector<buffer_t> gen_fused_fake_grads(const std::vector<int> &sizes)
{
    TRACE_SCOPE(__func__);
    int total_size = 0;
    int idx        = 0;
    std::string fused_name;
    for (const auto &size : sizes) {
        fused_name += fake_grad_name(idx++);
        total_size += size;
    }
    std::vector<buffer_t> grads;
    grads.push_back(buffer_t(fused_name, total_size));
    return grads;
}

template <bool async = true> class fake_minibatch_runner_t
{
    using ms = std::chrono::milliseconds;

    const bool _fake_train = false;
    const ms _fake_sleep_duration;

    template <typename buffer_t, typename Collective>
    void run_sync(std::vector<buffer_t> &grads, Collective &comm) const
    {
        for (auto &g : grads) {
            TRACE_SCOPE("run_sync::collective_all_reduce");
            collective_all_reduce(g.send_buf.data(), g.recv_buf.data(), g.count,
                                  g.name.c_str(), comm);
        }
    }

    template <typename buffer_t, typename Collective>
    void run_async(std::vector<buffer_t> &grads, Collective &comm) const
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

    template <typename buffer_t, typename Collective>
    void operator()(std::vector<buffer_t> &grads, Collective &comm) const
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

void log_estimated_speed(int n_batches, int batch_size, testing::duration_t d,
                         int cluster_size)
{
    const double img_per_sec = n_batches * batch_size / d.count();
    fprintf(stderr,
            "Img/sec %.2f per worker, Img/sec %.2f per cluster, np=%d\n",
            img_per_sec, img_per_sec * cluster_size, cluster_size);
}

class fake_trainer_t
{
    const int n_iters;
    const int step_per_iter;
    const int batch_size;

  public:
    fake_trainer_t(int n_iters, int step_per_iter, int batch_size)
        : n_iters(n_iters), step_per_iter(step_per_iter), batch_size(batch_size)
    {
    }

    template <typename buffer_t, typename Step, typename Collective>
    void operator()(const Step &minibatch, std::vector<buffer_t> &grads,
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
            if (comm.is_root()) {
                fprintf(stderr, "%02d after %d steps\n", comm.rank(), step);
            }
        }
        log_estimated_speed(n_iters * step_per_iter, batch_size,
                            testing::since(t0), comm.cluster_size());
    }
};

template <typename Collective, typename buffer_t = fake_cpu_buffer_t<float>,
          bool fuse_grads = true>
void run_experiment(const std::vector<int> &grad_sizes, Collective &comm)
{
    TRACE_SCOPE(__func__);
    const int batch_size       = 32;
    const double image_per_sec = 185;
    const int n_iters          = 11;
    const int step_per_iter    = 10;

    constexpr bool async = false;
    fake_minibatch_runner_t<async> minibatch(batch_size, image_per_sec);
    fake_trainer_t train(n_iters, step_per_iter, batch_size);
    auto grads = fuse_grads ? gen_fused_fake_grads<buffer_t>(grad_sizes)
                            : gen_fake_grads<buffer_t>(grad_sizes);
    train(minibatch, grads, comm);
}
