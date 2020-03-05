#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "collective_mpi_impl.hpp"
#include "fake_model.hpp"
#include "fake_trainer.hpp"
#include "testing.hpp"

DEFINE_TRACE_CONTEXTS;

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    std::string model;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-model") == 0) {
            if (i + 1 < argc) {
                model = argv[i + 1];
            } else {
                fprintf(stderr, "missing argument: %s\n", argv[i]);
                exit(1);
            }
        }
    }
    printf("model: %s\n", model.c_str());
    mpi_collective mpi(argc, argv);
    const auto grad_sizes = parameter_sizes(model, false);
    run_experiment(grad_sizes, mpi);
    return 0;
}
