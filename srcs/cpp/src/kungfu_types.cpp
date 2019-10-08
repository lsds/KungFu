#include <kungfu.h>
#include <kungfu_types.hpp>

const KungFu_Datatype KungFu_UINT8 = kungfu::type_encoder::value<uint8_t>();
const KungFu_Datatype KungFu_INT32 = kungfu::type_encoder::value<int32_t>();
const KungFu_Datatype KungFu_INT64 = kungfu::type_encoder::value<int64_t>();
const KungFu_Datatype KungFu_FLOAT16 =
    kungfu::type_encoder::value<kungfu::float16>();
const KungFu_Datatype KungFu_FLOAT  = kungfu::type_encoder::value<float>();
const KungFu_Datatype KungFu_DOUBLE = kungfu::type_encoder::value<double>();

uint32_t kungfu_type_size(KungFu_Datatype dt)
{
    static_assert(sizeof(kungfu::float16) == 2, "");
    return dtype_size(dtype(dt));
}

const KungFu_Op KungFu_MAX = kungfu::op_encoder::value<kungfu::op_max>();
const KungFu_Op KungFu_MIN = kungfu::op_encoder::value<kungfu::op_min>();
const KungFu_Op KungFu_SUM = kungfu::op_encoder::value<kungfu::op_sum>();

const KungFu_AllReduceStrategy KungFu_StarAllReduce   = star;
const KungFu_AllReduceStrategy KungFu_RingAllReduce   = ring;
const KungFu_AllReduceStrategy KungFu_CliqueAllReduce = clique;
const KungFu_AllReduceStrategy KungFu_TreeAllReduce   = tree;
