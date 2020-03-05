#include <algorithm>
#include <vector>

#include <mpi.h>
#include <nccl.h>

#include "collective.hpp"
#include "collective_kungfu_go_impl.hpp"
#include "collective_mpi_impl.hpp"
#include "collective_nccl_impl.hpp"
#include "cuda_vector.hpp"
#include "fake_trainer.hpp"
#include "resnet50_info.hpp"
#include "testing.hpp"

DEFINE_TRACE_CONTEXTS;

template <typename T> struct fake_gpu_buffer_t {
    using value_type = T;
    const std::string name;
    const int count;

    cuda_vector<T> send_buf;
    cuda_vector<T> recv_buf;

    fake_gpu_buffer_t(const std::string &name, int count)
        : name(name), count(count), send_buf(count), recv_buf(count)
    {
    }
};

constexpr size_t Mi = 1 << 20;

void simple_test(int size, nccl_collective &nccl)
{
    const int rank = nccl.rank();
    const int np   = nccl.cluster_size();

    printf("simple_test of size: %d Mi\n", (int)(size / Mi));
    int n = size / sizeof(int32_t);
    cuda_vector<int32_t> x(n);
    {
        std::vector<int32_t> v(n);
        std::fill(v.begin(), v.end(), rank);
        x.from_host(v.data());
    }
    cuda_vector<int32_t> y(n);
    nccl.all_reduce(x.data(), y.data(), n, "test-tensor");
    {
        const int s = np * (np - 1) / 2;
        std::vector<int32_t> v(n);
        y.to_host(v.data());
        for (int i = 0; i < n; i++) {
            if (v[i] != s) {
                fprintf(stderr,
                        "incorrect all_reduce result, expect %d, got %d", s,
                        v[i]);
                exit(1);
            }
        }
    }
    printf("simple_test result is correct\n");
}

template <typename Collective> int main1(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    Collective bootstrap(argc, argv);

    ncclUniqueId id;
    if (bootstrap.is_root()) {
        KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id);
    }
    bootstrap.template bcast<uint8_t>((uint8_t *)&id, sizeof(id), "nccl id");

    nccl_collective nccl(id, bootstrap.cluster_size(), bootstrap.rank());
    {
        TRACE_SCOPE("run simple tests");
        for (int i = 1; i < 10; ++i) { simple_test(i * Mi, nccl); }
        for (int i = 1; i < 10; ++i) { simple_test(i * 10 * Mi, nccl); }
    }
    printf("simple tests are done\n");

    const auto grad_sizes = resnet50_grad_sizes();
    run_experiment<nccl_collective, fake_gpu_buffer_t<float>>(grad_sizes, nccl);
    return 0;
}

int main(int argc, char *argv[])
{
    bool using_kungfu = not safe_getenv("KUNGFU_TEST_CLUSTER_SIZE").empty();
    if (using_kungfu) {
        return main1<kungfu_go_collective>(argc, argv);
    } else {
        return main1<mpi_collective>(argc, argv);
    }
}
