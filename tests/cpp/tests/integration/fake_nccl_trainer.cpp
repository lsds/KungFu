#include <mpi.h>
#include <nccl.h>

#include "collective.hpp"
#include "collective_kungfu_go_impl.hpp"
#include "collective_mpi_impl.hpp"
#include "collective_nccl_impl.hpp"
#include "cuda_vector.hpp"
#include "testing.hpp"

template <typename Collective> int main1(int argc, char *argv[])
{
    Collective bootstrap(argc, argv);

    ncclUniqueId id;
    if (bootstrap.is_root()) { ncclGetUniqueId(&id); }
    bootstrap.template bcast<uint8_t>((uint8_t *)&id, sizeof(id), "nccl id");

    nccl_collective nccl(id, bootstrap.cluster_size(), bootstrap.rank());

    using T     = float;
    const int n = 10;
    cuda_vector<T> x(n);
    cuda_vector<T> y(n);

    collective_all_reduce(x.data(), y.data(), n, "test-tensor", nccl);

    return 0;
}

int main(int argc, char *argv[])
{
    bool using_kungfu = not safe_getenv("KUNGFU_SELF_RANK").empty();
    if (using_kungfu) {
        return main1<kungfu_go_collective>(argc, argv);
    } else {
        return main1<mpi_collective>(argc, argv);
    }
}
