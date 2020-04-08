#include <kungfu/peer.hpp>
#include <libkungfu-comm.h>

kungfu_world::kungfu_world()
{
    const int err = GoKungfuInit();
    if (err) {
        fprintf(stderr, "%s failed\n", "GoKungfuInit");
        exit(1);
    }
}

kungfu_world::~kungfu_world() { GoKungfuFinalize(); }

uint64_t kungfu_world::Uid() const { return GoKungfuUID(); }

int kungfu_world::Save(const char *name, const void *buf, int count,
                       KungFu_Datatype dtype)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), dtype, nullptr);
}

int kungfu_world::Save(const char *name, const void *buf, int count,
                       KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuSave(const_cast<char *>(name), const_cast<void *>(buf),
                        GoInt(count), dtype, new CallbackWrapper(done));
}

int kungfu_world::Save(const char *version, const char *name, const void *buf,
                       int count, KungFu_Datatype dtype)
{
    return GoKungfuSaveVersion(
        const_cast<char *>(version), const_cast<char *>(name),
        const_cast<void *>(buf), GoInt(count), dtype, nullptr);
}

int kungfu_world::Save(const char *version, const char *name, const void *buf,
                       int count, KungFu_Datatype dtype,
                       const DoneCallback &done)
{
    return GoKungfuSaveVersion(const_cast<char *>(version),
                               const_cast<char *>(name),
                               const_cast<void *>(buf), GoInt(count), dtype,
                               new CallbackWrapper(done));
}

int kungfu_world::Request(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, nullptr);
}

int kungfu_world::Request(int destRank, const char *name, void *buf, int count,
                          KungFu_Datatype dtype, const DoneCallback &done)
{
    return GoKungfuRequest(destRank, const_cast<char *>(name), buf,
                           GoInt(count), dtype, new CallbackWrapper(done));
}

int kungfu_world::Request(int rank, const char *version, const char *name,
                          void *buf, int count, KungFu_Datatype dtype)
{
    return GoKungfuRequestVersion(rank, const_cast<char *>(version),
                                  const_cast<char *>(name), buf, GoInt(count),
                                  dtype, nullptr);
}

int kungfu_world::Request(int rank, const char *version, const char *name,
                          void *buf, int count, KungFu_Datatype dtype,
                          const DoneCallback &done)
{
    return GoKungfuRequestVersion(rank, const_cast<char *>(version),
                                  const_cast<char *>(name), buf, GoInt(count),
                                  dtype, new CallbackWrapper(done));
}

// monitoring APIs
int kungfu_world::GetPeerLatencies(float *recvbuf, int recv_count)
{
    return GoKungfuGetPeerLatencies(recvbuf, recv_count, KungFu_FLOAT);
}

// control APIs
int kungfu_world::ResizeClusterFromURL(bool *changed, bool *keep)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    return GoKungfuResizeClusterFromURL(reinterpret_cast<char *>(changed),
                                        reinterpret_cast<char *>(keep));
}

int kungfu_world::ProposeNewSize(int new_size)
{
    return GoKungfuProposeNewSize(GoInt(new_size));
}
