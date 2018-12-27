#include <stdexcept>
#include <string>

#include <kungfu.h>
#include <kungfu_types.hpp>

KungFu_Datatype KungFu_INT32  = kungfu::type_encoder::value<int32_t>();
KungFu_Datatype KungFu_FLOAT  = kungfu::type_encoder::value<float>();
KungFu_Datatype KungFu_DOUBLE = kungfu::type_encoder::value<double>();

uint32_t kungfu_type_size(KungFu_Datatype dtype)
{
    switch (dtype) {
    case kungfu::type_encoder::value<int32_t>():
        return sizeof(int32_t);
    case kungfu::type_encoder::value<float>():
        return sizeof(float);
    case kungfu::type_encoder::value<double>():
        return sizeof(double);
    default:
        throw std::invalid_argument("kungfu doesn't support dtype: " +
                                    std::to_string(dtype));
    }
}

KungFu_Op KungFu_MAX = kungfu::op_encoder::value<kungfu::op_max>();
KungFu_Op KungFu_MIN = kungfu::op_encoder::value<kungfu::op_min>();
KungFu_Op KungFu_SUM = kungfu::op_encoder::value<kungfu::op_sum>();

KungFu_AllReduceAlgo KungFu_SimpleAllReduce        = 0;
KungFu_AllReduceAlgo KungFu_RingAllReduce          = 1;
KungFu_AllReduceAlgo KungFu_FullSymmetricAllReduce = 2;
KungFu_AllReduceAlgo KungFu_TreeAllReduce          = 3;
