#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>

#include <cuda_runtime.h>
#include <nccl.h>

#include "testing.hpp"
#include <kungfu/cuda/stream.hpp>
#include <kungfu/utils/error_checker.hpp>

using kungfu::cuda_checker;

struct show_nccl_error {
    std::string operator()(ncclResult_t err) const
    {
        return ncclGetErrorString(err);
    }
};

using nccl_checker = error_checker<ncclResult_t, ncclSuccess, show_nccl_error>;

template <typename T>
struct nccl_type;
template <>
struct nccl_type<int32_t> {
    static auto value() { return ncclInt32; }
};
template <>
struct nccl_type<float> {
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
        bool using_kungfu = not safe_getenv("KUNGFU_TEST_CLUSTER_SIZE").empty();
        if (using_kungfu) {
            KUNGFU_CHECK(cuda_checker) << cudaSetDevice(0);
            printf("cuda device selected to %d\n", 0);
        } else {
            KUNGFU_CHECK(cuda_checker) << cudaSetDevice(rank);
            printf("cuda device selected to %d\n", rank);
        }
        KUNGFU_CHECK(nccl_checker)
            << ncclCommInitRank(&comm, cluster_size, id, rank);
        printf("nccl inited: %d/%d.\n", rank, cluster_size);
    }

    ~nccl_collective()
    {
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
        KUNGFU_CHECK(cuda_checker) << cudaStreamCreate(&stream);
        KUNGFU_CHECK(nccl_checker)
            << ncclAllReduce(send_buf, recv_buf, count, nccl_type<T>::value(),
                             ncclSum, comm, stream);
        KUNGFU_CHECK(cuda_checker) << cudaStreamSynchronize(stream);
        KUNGFU_CHECK(cuda_checker) << cudaStreamDestroy(stream);
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
