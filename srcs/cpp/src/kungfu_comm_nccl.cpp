#include <kungfu_comm.hpp>

#include <nccl.h>

#include "cuda_helper.hpp"
#include "error_checker.hpp"

struct show_nccl_error {
    std::string operator()(ncclResult_t err) const
    {
        return ncclGetErrorString(err);
    }
};

using nccl_checker = error_checker<ncclResult_t, ncclSuccess, show_nccl_error>;

template <typename T> struct nccl_type;
template <> struct nccl_type<int32_t> {
    static ncclDataType_t value() { return ncclInt32; }
};
template <> struct nccl_type<float> {
    static ncclDataType_t value() { return ncclFloat; }
};

namespace kungfu
{

ncclDataType_t to_nccl_type(const KungFu_Datatype dtype)
{
    switch (dtype) {
    case type_encoder::value<int32_t>():
        return nccl_type<int32_t>::value();
    case type_encoder::value<float>():
        return nccl_type<float>::value();
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

class nccl_comm : public sync_communicator
{
    ncclComm_t comm;
    const int _cluster_size;
    const int _rank;
    const int _root;

  public:
    nccl_comm(ncclUniqueId id, int cluster_size, int rank)
        : _cluster_size(cluster_size), _rank(rank), _root(0)
    {
        KUNGFU_CHECK(cuda_checker) << cudaSetDevice(0);
        KUNGFU_CHECK(nccl_checker)
            << ncclCommInitRank(&comm, cluster_size, id, rank);
    }

    ~nccl_comm() { ncclCommDestroy(comm); }

    void bcast(const void *send_buf, void *recv_buf, size_t count,
               KungFu_Datatype dtype)
    {
        cuda_stream stream;
        // https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/colls.html#ncclbroadcast
        KUNGFU_CHECK(nccl_checker)
            << ncclBroadcast(send_buf, recv_buf, count, to_nccl_type(dtype),
                             _root, comm, stream);
        stream.sync();
    }

    void reduce(const void *send_buf, void *recv_buf, size_t count,
                KungFu_Datatype dtype)
    {
        cuda_stream stream;
        // https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/colls.html#ncclreduce
        KUNGFU_CHECK(nccl_checker)
            << ncclReduce(send_buf, recv_buf, count, to_nccl_type(dtype),
                          ncclSum, _root, comm, stream);
        stream.sync();
    }

    void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                    KungFu_Datatype dtype)
    {
        cuda_stream stream;
        // https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/colls.html#ncclallreduce
        KUNGFU_CHECK(nccl_checker)
            << ncclAllReduce(send_buf, recv_buf, count, to_nccl_type(dtype),
                             ncclSum, comm, stream);
        stream.sync();
    }
};

communicator *new_local_nccl_comm(kungfu_world &bootstrap)
{
    ncclUniqueId id;
    const int root = 0;
    const int rank = bootstrap.Rank();
    if (rank == root) { KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id); }
    bootstrap.Broadcast(&id, &id, sizeof(id), type_encoder::value<uint8_t>(),
                        "nccl id");
    return new nccl_comm(id, bootstrap.ClusterSize(), rank);
}
}  // namespace kungfu
