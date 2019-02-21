#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "collective_mpi_impl.hpp"
#include "fake_trainer.hpp"
#include "resnet50_info.hpp"
#include "testing.hpp"

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    mpi_collective mpi(argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const bool is_root         = rank == 0;
    const int batch_size       = 32;
    const double image_per_sec = 185;
    const int n_iters          = 11;
    const int step_per_iter    = 10;

    constexpr bool async = false;
    fake_minibatch_runner_t<async> minibatch(batch_size, image_per_sec);
    fake_trainer_t train(is_root, n_iters, step_per_iter, batch_size);

    bool fuse_grads  = true;
    using T          = float;
    const auto sizes = resnet50_grad_sizes();
    auto grads =
        fuse_grads ? gen_fused_fake_grads<T>(sizes) : gen_fake_grads<T>(sizes);
    train(minibatch, grads, mpi);
    return 0;
}
