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

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const bool is_root    = rank == 0;
    const auto grad_sizes = resnet50_grad_sizes();
    run_experiment(is_root, world_size, grad_sizes, mpi);
    return 0;
}
