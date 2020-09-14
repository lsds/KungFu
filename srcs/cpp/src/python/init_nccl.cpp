#include <kungfu.h>
#include <kungfu/nccl/helper.hpp>
#include <kungfu/ncclv2/helper.hpp>
#include <kungfu/python/init.h>

std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;
std::unique_ptr<kungfu::NCCLHelper_V2> _default_nccl_helper_v2;

void kungfu_python_init_nccl()
{
    // _default_nccl_helper.reset(new kungfu::NCCLHelper);
    _default_nccl_helper_v2.reset(
        new kungfu::NCCLHelper_V2(_default_peer.get()));
}

void kungfu_python_finialize_nccl()
{
    // _default_nccl_helper.reset(nullptr);
    _default_nccl_helper_v2.reset(nullptr);
}
