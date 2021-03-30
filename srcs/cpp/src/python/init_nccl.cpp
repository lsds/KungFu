#include <kungfu.h>
#include <kungfu/nccl/helper.hpp>
#include <kungfu/python/c_api.h>

std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;

void kungfu_python_init_nccl()
{
    _default_nccl_helper.reset(new kungfu::NCCLHelper);
}

void kungfu_python_finialize_nccl() { _default_nccl_helper.reset(nullptr); }
