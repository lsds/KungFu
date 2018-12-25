#include <kungfu.h>
#include <kungfu_types.hpp>

KungFu_Datatype KungFu_INT = kungfu::type_encoder::value<int>();
KungFu_Datatype KungFu_FLOAT = kungfu::type_encoder::value<float>();
KungFu_Datatype KungFu_DOUBLE = kungfu::type_encoder::value<double>();

KungFu_Op KungFu_MAX = kungfu::op_encoder::value<kungfu::op_max>();
KungFu_Op KungFu_MIN = kungfu::op_encoder::value<kungfu::op_min>();
KungFu_Op KungFu_SUM = kungfu::op_encoder::value<kungfu::op_sum>();
