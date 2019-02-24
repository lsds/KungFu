#pragma once
#include <functional>
#include <iostream>

#include <cuda_runtime.h>
#include <nccl.h>

struct check_cuda {
    const check_cuda &operator<<(cudaError_t error) const
    {
        if (error != cudaSuccess) {
            fprintf(stderr, "cuda error %d\n", error);
            perror(cudaGetErrorString(error));
            exit(1);
        }
        return *this;
    }
};

struct check_nccl {
    const check_nccl &operator<<(ncclResult_t error) const
    {
        if (error != ncclSuccess) {
            fprintf(stderr, "nccl error %d\n", error);
            perror(ncclGetErrorString(error));
            exit(1);
        }
        return *this;
    }
};

template <typename T> struct nccl_type;
template <> struct nccl_type<float> {
    static auto value() { return ncclFloat; }
};

class nccl_collective
{
    ncclComm_t comm;
    const int _rank;
    const int _cluster_size;

  public:
    nccl_collective(ncclUniqueId id, int cluster_size, int rank)
        : _rank(rank), _cluster_size(cluster_size)
    {
        check_nccl() << ncclCommInitRank(&comm, cluster_size, id, rank);
        printf("nccl inited: %d/%d.\n", rank, cluster_size);
        check_cuda() << cudaSetDevice(rank);
    }

    ~nccl_collective()
    {
        printf("before nccl destroyed: %d/%d.\n", _rank, _cluster_size);
        ncclCommDestroy(comm);
        printf("nccl destroyed: %d/%d.\n", _rank, _cluster_size);
    }

    bool is_root() const { return _rank == 0; }

    int rank() const { return _rank; }

    int cluster_size() const { return _cluster_size; }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char * /* FIXME: ignored */)
    {
        cudaStream_t stream;
        check_cuda() << cudaStreamCreate(&stream);

        check_nccl() << ncclAllReduce(send_buf, recv_buf, count,
                                      nccl_type<T>::value(), ncclSum, comm,
                                      stream);
        // printf("ncclAllReduce done.\n");

        check_cuda() << cudaStreamSynchronize(stream);
        // printf("cudaStreamSynchronize done.\n");

        check_cuda() << cudaStreamDestroy(stream);
        // printf("cudaStreamDestroy done.\n");
    }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char *name, std::function<void()> done)
    {
        // FIXME: not supported
        std::cerr << "nccl_collective::all_reduce<async> is not implemted"
                  << std::endl;
        done();
    }
};
