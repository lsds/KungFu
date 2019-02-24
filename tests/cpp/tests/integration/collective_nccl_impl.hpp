#pragma once
#include <functional>
#include <iostream>

#include <nccl.h>

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
        ncclCommInitRank(&comm, cluster_size, id, rank);
        printf("nccl inited: %d/%d.\n", rank, cluster_size);
    }

    ~nccl_collective()
    {
        printf("before nccl destroyed: %d/%d.\n", _rank, _cluster_size);
        ncclCommDestroy(comm);
        printf("nccl destroyed: %d/%d.\n", _rank, _cluster_size);
    }

    bool is_root() const { return _rank == 0; }

    int cluster_size() const { return _cluster_size; }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char * /* FIXME: ignored */)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        ncclAllReduce(send_buf, recv_buf, count, nccl_type<T>::value(), ncclSum,
                      comm, stream);
        // printf("ncclAllReduce done.\n");

        cudaStreamSynchronize(stream);
        // printf("cudaStreamSynchronize done.\n");

        cudaStreamDestroy(stream);
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
