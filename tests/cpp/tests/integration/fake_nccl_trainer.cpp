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

template <typename T> struct fake_gpu_buffer_t {
    using value_type = T;
    const std::string name;
    const int count;

    cuda_vector<T> send_buf;
    cuda_vector<T> recv_buf;

    fake_gpu_buffer_t(const std::string &name, int count)
        : name(name), count(count), send_buf(count), recv_buf(count)
    {
        printf("fake_gpu_buffer_t of count %d, size %d\n", count,
               (int)(count * sizeof(T)));
    }
};

constexpr size_t Mi = 1 << 10;

void simple_test(int size)
{
    printf("simple_test of size: %d Mi\n", (int)(size / Mi));
    int n = size / sizeof(float);
    cuda_vector<float> x(n);
    cuda_vector<float> y(n);
    nccl.all_reduce(x.data(), y.data(), n, "test-tensor");
}

template <typename Collective> int main1(int argc, char *argv[])
{
    Collective bootstrap(argc, argv);

    ncclUniqueId id;
    if (bootstrap.is_root()) { CHECK(nccl_checker) << ncclGetUniqueId(&id); }
    bootstrap.template bcast<uint8_t>((uint8_t *)&id, sizeof(id), "nccl id");

    nccl_collective nccl(id, bootstrap.cluster_size(), bootstrap.rank());

    for (int i = 1; i < 10; ++i) { simple_test(i * Mi); }
    for (int i = 1; i < 10; ++i) { simple_test(i * 10 * Mi); }
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
