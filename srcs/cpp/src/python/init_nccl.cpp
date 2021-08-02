#include <kungfu.h>
#include <kungfu/nccl/helper.hpp>
#include <kungfu/python/c_api.h>

void kungfu_python_init_nccl() { kungfu::NCCLHelper::GetDefault(true); }

void kungfu_python_finialize_nccl()
{
    auto &p = kungfu::NCCLHelper::GetDefault();
    p.reset(nullptr);
}
