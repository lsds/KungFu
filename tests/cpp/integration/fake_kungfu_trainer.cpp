#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "collective_kungfu_go_impl.hpp"
#include "fake_trainer.hpp"
#include "resnet50_info.hpp"
#include "testing.hpp"

DEFINE_TRACE_CONTEXTS;

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    kungfu_go_collective kungfu;
    const auto grad_sizes = resnet50_grad_sizes();
    run_experiment(grad_sizes, kungfu);
    return 0;
}
