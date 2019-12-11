#include <algorithm>
#include <cstdio>
#include <sstream>
#include <vector>

#include <kungfu/nccl/gpu_collective.hpp>
#include <kungfu/utils/cuda_helper.hpp>

#include <nccl.h>

struct show_nccl_error {
    std::string operator()(ncclResult_t err) const
    {
        return ncclGetErrorString(err);
    }
};

using nccl_checker = error_checker<ncclResult_t, ncclSuccess, show_nccl_error>;

void kungfu_show_cuda_version()
{
    int driverVersion;
    KUNGFU_CHECK(cuda_checker) << cudaDriverGetVersion(&driverVersion);
    printf("CUDA Driver Veresion: %d\n", driverVersion);

    int runtimeVersion;
    KUNGFU_CHECK(cuda_checker) << cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Runtime Veresion: %d\n", runtimeVersion);
}

void kungfu_show_nccl_version()
{
    int version;
    KUNGFU_CHECK(nccl_checker) << ncclGetVersion(&version);
    printf("NCCL Version: %d\n", version);
}

template <typename T> struct nccl_type;
template <> struct nccl_type<int32_t> {
    static ncclDataType_t value() { return ncclInt32; }
};
template <> struct nccl_type<kungfu::float16> {
    static ncclDataType_t value() { return ncclFloat16; }
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
    case type_encoder::value<kungfu::float16>():
        return nccl_type<kungfu::float16>::value();
    case type_encoder::value<float>():
        return nccl_type<float>::value();
    default:
        // TODO: add more types
        throw std::invalid_argument("unsupported dtype");
    }
}

class gpu_collective_nccl : public gpu_collective
{
    ncclComm_t comm;
    const int _rank;
    const int _cluster_size;

    cuda_stream _stream;

  public:
    gpu_collective_nccl(ncclUniqueId id, int cluster_size, int rank)
        : _rank(rank), _cluster_size(cluster_size)
    {
        KUNGFU_CHECK(nccl_checker)
            << ncclCommInitRank(&comm, cluster_size, id, rank);
    }

    ~gpu_collective_nccl()
    {
        // KUNGFU_CHECK(nccl_checker) <<
        ncclCommDestroy(comm);
    }

    void all_reduce(const void *send_buf, void *recv_buf, size_t count,
                    KungFu_Datatype dtype)
    {
        // https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/colls.html#ncclallreduce
        KUNGFU_CHECK(nccl_checker)
            << ncclAllReduce(send_buf, recv_buf, count, to_nccl_type(dtype),
                             ncclSum, comm, _stream);
        _stream.sync();
    }
};

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> parts;
    std::string part;
    std::istringstream ss(s);
    while (std::getline(ss, part, sep)) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

std::vector<int> parse_cuda_visible_devices(const std::string &val)
{
    const auto parts = split(val, ',');
    std::vector<int> devs;
    for (const auto &p : parts) {
        const int idx = std::stoi(p);
        if (idx >= 0) { devs.push_back(idx); }
    }
    return devs;
}

void setDevice()
{
    int dev = 0;
    {
        const char *ptr = std::getenv("KUNGFU_CUDA_VISIBLE_DEVICES");
        if (ptr != nullptr) { dev = std::stoi(ptr); }
    }
    {
        const char *ptr = std::getenv("CUDA_VISIBLE_DEVICES");
        if (ptr != nullptr) {
            const auto devs = parse_cuda_visible_devices(ptr);
            int idx = std::find(devs.begin(), devs.end(), dev) - devs.begin();
            if (idx == static_cast<int>(devs.size())) { idx = -1; }
            KUNGFU_CHECK(cuda_checker) << cudaSetDevice(idx);
        } else {
            KUNGFU_CHECK(cuda_checker) << cudaSetDevice(dev);
        }
    }
}

gpu_collective *new_gpu_collective(kungfu_world &world)
{
    ncclUniqueId id;
    const int root = 0;
    const int rank = world.Rank();

    setDevice();

    if (rank == root) { KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id); }
    world.Broadcast(&id, &id, sizeof(id), type_encoder::value<uint8_t>(),
                    "nccl id");
    return new gpu_collective_nccl(id, world.ClusterSize(), rank);
}
}  // namespace kungfu
