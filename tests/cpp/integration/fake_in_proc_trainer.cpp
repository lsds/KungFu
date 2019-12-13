#include <vector>

#include "fake_trainer.hpp"
#include "resnet50_info.hpp"
#include "testing.hpp"

DEFINE_TRACE_CONTEXTS;

template <typename buffer_t>
void in_proc_all_reduce(int np, std::vector<std::vector<buffer_t>> &node_grads,
                        int n_grads)
{
    for (int j = 0; j < n_grads; ++j) {
        TRACE_SCOPE("run_sync::collective_all_reduce");
        if (np == 1) {
            node_grads[0][j].recv_self();
        } else {
            for (int i = 1; i < np; ++i) {
                node_grads[i][j].recv_onto(node_grads[i - 1][j]);
            }
            for (int i = np - 1; i > 0; --i) {
                node_grads[i - 1][j].recv_into(node_grads[i][j]);
            }
        }
    }
}

template <typename buffer_t>
bool test_in_proc_all_reduce(int np, const std::vector<int> &grad_sizes)
{
    TRACE_SCOPE(__func__);
    const auto grads  = gen_fused_fake_grads<buffer_t>(grad_sizes);
    const int n_grads = grads.size();
    std::vector<std::vector<buffer_t>> node_grads;
    for (int i = 0; i < np; ++i) { node_grads.push_back(grads); }

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < n_grads; ++j) { node_grads[i][j].reset(i + 1); }
    }
    in_proc_all_reduce(np, node_grads, n_grads);
    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < n_grads; ++j) {
            const typename buffer_t::value_type result = np * (np + 1) / 2;
            if (!node_grads[i][j].check(i + 1, result)) { return false; }
        }
    }
    return true;
}

template <typename buffer_t>
void train_in_proc_all_reduce(int np, const std::vector<int> &grad_sizes)
{
    TRACE_SCOPE(__func__);
    const auto grads  = gen_fused_fake_grads<buffer_t>(grad_sizes);
    const int n_grads = grads.size();

    std::vector<std::vector<buffer_t>> node_grads;
    for (int i = 0; i < np; ++i) { node_grads.push_back(grads); }

    const int batch_size    = 32;
    const int n_iters       = 11;
    const int step_per_iter = 10;

    const auto t0 = testing::now();
    for (int i = 0; i < n_iters; ++i) {
        for (int j = 0; j < step_per_iter; ++j) {
            TRACE_SCOPE("mini batch");
            in_proc_all_reduce(np, node_grads, n_grads);
        }
    }
    log_estimated_speed(n_iters * step_per_iter, batch_size, testing::since(t0),
                        np);
}

int getTestClusterSize()
{
    return std::stoi(safe_getenv("KUNGFU_TEST_CLUSTER_SIZE"));
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    const int np          = getTestClusterSize();
    const auto grad_sizes = resnet50_grad_sizes();
    if (!test_in_proc_all_reduce<fake_cpu_buffer_t<int>>(np, grad_sizes)) {
        fprintf(stderr, "invalid in_proc_all_reduce result for np=%d\n", np);
        return 1;
    }
    train_in_proc_all_reduce<fake_cpu_buffer_t<float>>(np, grad_sizes);
    return 0;
}
