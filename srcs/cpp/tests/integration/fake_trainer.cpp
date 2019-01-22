#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "resnet50_info.hpp"
#include "testing.hpp"

bool is_root = getSelfRank() == 0;

template <typename T> struct fake_grad_t {
    std::string name;
    int count;
    KungFu_Datatype dtype;
    std::vector<T> send_buf;
    std::vector<T> recv_buf;

    fake_grad_t(const std::string &name, int count)
        : name(name), count(count), dtype(kungfu::type_encoder::value<T>()),
          send_buf(count), recv_buf(count)
    {
    }
};

template <typename T> using grad_list_t = std::vector<fake_grad_t<T>>;

template <typename T> grad_list_t<T> gen_fake_grads()
{
    TRACE_SCOPE(__func__);
    const auto sizes = resnet50_grad_sizes();
    const int n      = sizes.size();
    grad_list_t<T> grads;
    for (int i = 0; i < n; ++i) {
        fake_grad_t<T> g(resnet50_grad_name(i), sizes[i]);
        grads.push_back(g);
    }
    return grads;
}

template <typename T> grad_list_t<T> gen_fused_fake_grads()
{
    TRACE_SCOPE(__func__);
    const auto sizes = resnet50_grad_sizes();
    const int n      = sizes.size();
    int total_size   = 0;
    std::string name;
    for (int i = 0; i < n; ++i) {
        name += resnet50_grad_name(i);
        total_size += sizes[i];
    }
    grad_list_t<T> grads;
    fake_grad_t<T> g(name, total_size);
    grads.push_back(g);
    return grads;
}

template <bool async = true> class fake_minibatch_runner_t
{
    template <typename T> void run_sync(grad_list_t<T> &grads) const
    {
        for (auto &g : grads) {
            TRACE_SCOPE("run_sync::KungfuNegotiate");
            KungfuNegotiate(g.send_buf.data(), g.recv_buf.data(), g.count,
                            g.dtype, KungFu_SUM, g.name.c_str());
        }
    }

    template <typename T> void run_async(grad_list_t<T> &grads) const
    {
        std::vector<Waiter> waiters(grads.size());
        int idx = 0;
        for (auto &g : grads) {
            KungfuNegotiate(g.send_buf.data(), g.recv_buf.data(), g.count,
                            g.dtype, KungFu_SUM, g.name.c_str(),
                            [i = idx++, &waiters] { waiters[i].done(); });
        }
        for (auto &w : waiters) { w.wait(); }
    }

  public:
    template <typename T> void operator()(grad_list_t<T> &grads) const
    {
        if (async) {
            run_async(grads);
        } else {
            run_sync(grads);
        }
    }
};

class fake_trainer_t
{
    const int n_iters;
    const int step_per_iter;

  public:
    fake_trainer_t(int n_iters, int step_per_iter)
        : n_iters(n_iters), step_per_iter(step_per_iter)
    {
    }

    template <typename T, typename Step>
    void operator()(const Step &train_step, grad_list_t<T> &grads)
    {
        int step = 0;
        for (int i = 0; i < n_iters; ++i) {
            for (int j = 0; j < step_per_iter; ++j) {
                ++step;
                TRACE_SCOPE("train step");
                train_step(grads);
            }
            if (is_root) { fprintf(stderr, "after %d steps\n", step); }
        }
    }
};

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    kungfu_world _kungfu_world;

    const int n_iters       = 11;
    const int step_per_iter = 10;

    constexpr bool async = false;
    fake_minibatch_runner_t<async> minibatch;
    fake_trainer_t train(n_iters, step_per_iter);

    bool fuse_grads = true;
    using T         = float;
    auto grads = fuse_grads ? gen_fused_fake_grads<T>() : gen_fake_grads<T>();

    train(minibatch, grads);

    return 0;
}
