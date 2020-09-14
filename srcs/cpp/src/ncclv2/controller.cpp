#include <iostream>

#include <kungfu/cuda/stream.hpp>
#include <kungfu/ncclv2/controller.hpp>
#include <kungfu/python/init.h>
#include <kungfu/utils/error_checker.hpp>

#include <nccl.h>

namespace kungfu
{
struct show_nccl_error {
    std::string operator()(ncclResult_t err) const
    {
        return std::to_string(static_cast<int>(err)) + ": " +
               ncclGetErrorString(err);
    }
};

using nccl_checker = error_checker<ncclResult_t, ncclSuccess, show_nccl_error>;

template <typename T>
struct nccl_type;
template <>
struct nccl_type<int32_t> {
    static ncclDataType_t value() { return ncclInt32; }
};
template <>
struct nccl_type<kungfu::float16> {
    static ncclDataType_t value() { return ncclFloat16; }
};
template <>
struct nccl_type<float> {
    static ncclDataType_t value() { return ncclFloat; }
};

ncclDataType_t to_nccl_type_v2(const KungFu_Datatype dtype)
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

// NCCLComm wraps ncclComm_t
class NCCLComm
{
    const int root_;
    ncclComm_t comm_;
    CudaStream stream_;

  public:
    NCCLComm(ncclUniqueId id, int rank, int size, int root) : root_(root)
    {
        std::cout << __func__ << " rank: " << rank << ", size: " << size
                  << std::endl;
        KUNGFU_CHECK(nccl_checker) << ncclCommInitRank(&comm_, size, id, rank);
    }

    ~NCCLComm()
    {
        DBG(__func__);
        KUNGFU_CHECK(nccl_checker) << ncclCommDestroy(comm_);
    }

    void Reduce(Workspace w)
    {
        KUNGFU_CHECK(nccl_checker) << ncclReduce(
            w.sendbuf, w.recvbuf, w.count, to_nccl_type_v2(w.dtype), ncclSum,
            root_, comm_, stream_);
        stream_.sync();
    }

    void Broadcast(Workspace w)
    {
        KUNGFU_CHECK(nccl_checker)
            << ncclBroadcast(w.sendbuf, w.recvbuf, w.count,
                             to_nccl_type_v2(w.dtype), root_, comm_, stream_);
        stream_.sync();
    }

    void AllReduce(Workspace w)
    {
        KUNGFU_CHECK(nccl_checker)
            << ncclAllReduce(w.sendbuf, w.recvbuf, w.count,
                             to_nccl_type_v2(w.dtype), ncclSum, comm_, stream_);
        stream_.sync();
    }
};

class NCCLControllerImpl : public NCCLController_V2
{
    Peer *peer_;
    const KungFu_NCCLScope scope_;

    // NCCLComm should be constructed in the dedicated thread of NCCLScheduler
    std::unique_ptr<NCCLComm> comm_;

    void InitGlobal();
    void InitLocal();

  public:
    NCCLControllerImpl(Peer *peer, KungFu_NCCLScope scope);
    ~NCCLControllerImpl();

    void Init() override;
    void InitOnce() override;
    void Reduce(Workspace w) override;
    void Broadcast(Workspace w) override;
    void AllReduce(Workspace w) override;
};

NCCLControllerImpl::NCCLControllerImpl(Peer *peer, KungFu_NCCLScope scope)
    : peer_(peer), scope_(scope)
{
    DBG(__func__);
}

NCCLControllerImpl::~NCCLControllerImpl()
{
    DBG(__func__);  //
}

void NCCLControllerImpl::InitLocal()
{
    LOG_THREAD(__func__);
    ncclUniqueId id;
    const int root = 0;
    const int rank = peer_->LocalRank();
    KUNGFU_CHECK(cuda_checker) << cudaSetDevice(kungfu_get_cuda_index());
    if (rank == root) { KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id); }
    peer_->LocalBroadcast(&id, &id, sizeof(id), type_encoder::value<uint8_t>(),
                          "local nccl id");
    comm_.reset(new NCCLComm(id, rank, peer_->LocalSize(), root));
}

void NCCLControllerImpl::InitGlobal()
{
    LOG_THREAD(__func__);
    ncclUniqueId id;
    const int root = 0;
    const int rank = peer_->Rank();
    KUNGFU_CHECK(cuda_checker) << cudaSetDevice(kungfu_get_cuda_index());
    if (rank == root) { KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id); }
    peer_->Broadcast(&id, &id, sizeof(id), type_encoder::value<uint8_t>(),
                     "nccl id");
    comm_.reset(new NCCLComm(id, rank, peer_->Size(), root));
}

void NCCLControllerImpl::Init()
{
    LOG_THREAD(__func__);
    if (scope_ == KungFu_NCCL_LOCAL) {
        InitLocal();
    } else {
        InitGlobal();
    }
}

void NCCLControllerImpl::InitOnce()
{
    // LOG_THREAD(__func__);
    if (comm_.get() == nullptr) { Init(); }
}

void NCCLControllerImpl::Reduce(Workspace w) { comm_->Reduce(w); }
void NCCLControllerImpl::Broadcast(Workspace w) { comm_->Broadcast(w); }
void NCCLControllerImpl::AllReduce(Workspace w) { comm_->AllReduce(w); }

NCCLController_V2 *NCCLController_V2::Create(Peer *peer, KungFu_NCCLScope scope)
{
    return new NCCLControllerImpl(peer, scope);
}
}  // namespace kungfu
