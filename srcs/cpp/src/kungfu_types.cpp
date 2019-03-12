#include <stdexcept>
#include <string>

#include <kungfu.h>
#include <kungfu_types.hpp>

const KungFu_Datatype KungFu_UINT8  = kungfu::type_encoder::value<uint8_t>();
const KungFu_Datatype KungFu_INT32  = kungfu::type_encoder::value<int32_t>();
const KungFu_Datatype KungFu_INT64  = kungfu::type_encoder::value<int64_t>();
const KungFu_Datatype KungFu_FLOAT  = kungfu::type_encoder::value<float>();
const KungFu_Datatype KungFu_DOUBLE = kungfu::type_encoder::value<double>();

uint32_t kungfu_type_size(KungFu_Datatype dtype)
{
    switch (dtype) {
    case kungfu::type_encoder::value<uint8_t>():
        return sizeof(uint8_t);
    case kungfu::type_encoder::value<int32_t>():
        return sizeof(int32_t);
    case kungfu::type_encoder::value<int64_t>():
        return sizeof(int64_t);
    case kungfu::type_encoder::value<float>():
        return sizeof(float);
    case kungfu::type_encoder::value<double>():
        return sizeof(double);
    default:
        throw std::invalid_argument("kungfu doesn't support dtype: " +
                                    std::to_string(dtype));
    }
}

const KungFu_Op KungFu_MAX = kungfu::op_encoder::value<kungfu::op_max>();
const KungFu_Op KungFu_MIN = kungfu::op_encoder::value<kungfu::op_min>();
const KungFu_Op KungFu_SUM = kungfu::op_encoder::value<kungfu::op_sum>();

const KungFu_AllReduceAlgo KungFu_StarAllReduce   = 0;
const KungFu_AllReduceAlgo KungFu_RingAllReduce   = 1;
const KungFu_AllReduceAlgo KungFu_CliqueAllReduce = 2;
const KungFu_AllReduceAlgo KungFu_TreeAllReduce   = 3;
const KungFu_AllReduceAlgo KungFu_HybirdAllReduce = 4;
