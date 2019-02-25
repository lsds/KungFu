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
    const auto grad_sizes = resnet50_grad_sizes();
    run_experiment(grad_sizes, mpi);
    return 0;
}
