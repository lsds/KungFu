#include <kungfu/nccl/common.hpp>

namespace kungfu
{
const std::map<std::string, KungFu_NCCLScope> _nccl_scopes({
    {"global", KungFu_NCCL_GLOBAL},
    {"local", KungFu_NCCL_LOCAL},
});
}
