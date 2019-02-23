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

  public:
    nccl_collective(ncclUniqueId id, int cluster_size, int rank)
    {
        ncclCommInitRank(&comm, cluster_size, id, rank);
        printf("nccl inited.\n");
    }

    ~nccl_collective()
    {
        ncclCommDestroy(comm);
        printf("nccl destroyed.\n");
    }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char * /* FIXME: ignored */)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        ncclAllReduce(send_buf, recv_buf, count, nccl_type<T>::value(), ncclSum,
                      comm, stream);
        printf("ncclAllReduce done.\n");

        cudaStreamSynchronize(stream);
        printf("cudaStreamSynchronize done.\n");

        cudaStreamDestroy(stream);
        printf("cudaStreamDestroy done.\n");
    }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char *name, std::function<void()> done)
    {
        // FIXME: not supported
        std::cerr << "nccl_collective::all_reduce<async> is not implemted"
                  << std::endl;
    }
};
