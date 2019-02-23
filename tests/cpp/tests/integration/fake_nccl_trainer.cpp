#include <mpi.h>
#include <nccl.h>

#include "collective.hpp"
#include "collective_mpi_impl.hpp"
#include "collective_nccl_impl.hpp"
#include "cuda_vector.hpp"

int main(int argc, char *argv[])
{
    mpi_collective mpi(argc, argv);

    ncclUniqueId id;
    if (mpi.is_root()) { ncclGetUniqueId(&id); }
    mpi.bcast<char>((char *)&id, sizeof(id), "nccl id");

    nccl_collective nccl(id, mpi.cluster_size(), mpi.rank());

    using T     = float;
    const int n = 10;
    cuda_vector<T> x(n);
    cuda_vector<T> y(n);

    collective_all_reduce(x.data(), y.data(), n, "test-tensor", nccl);

    return 0;
}
