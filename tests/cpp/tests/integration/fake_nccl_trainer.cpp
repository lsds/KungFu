#include <mpi.h>
#include <nccl.h>

#include "cuda_vector.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int cluster_size;
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    printf("rank=%d/%d\n", rank, cluster_size);

    ncclUniqueId id;
    const int root = 0;
    if (rank == root) { ncclGetUniqueId(&id); }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, root, MPI_COMM_WORLD);

    ncclComm_t comm;
    ncclCommInitRank(&comm, cluster_size, id, rank);
    printf("nccl inited.\n");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    using T     = float;
    const int n = 10;
    cuda_vector<T> x(n);
    cuda_vector<T> y(n);

    ncclAllReduce((const void *)x.data(), y.data(), n, ncclFloat, ncclSum, comm,
                  stream);
    printf("ncclAllReduce done.\n");

    cudaStreamSynchronize(stream);
    printf("cudaStreamSynchronize done.\n");

    cudaStreamDestroy(stream);
    printf("cudaStreamDestroy done.\n");

    ncclCommDestroy(comm);
    printf("nccl destroyed.\n");

    MPI_Finalize();
    return 0;
}
