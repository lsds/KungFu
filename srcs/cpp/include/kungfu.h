#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int KungFu_Datatype;

extern KungFu_Datatype KungFu_INT;
// extern KungFu_Datatype KungFu_INT8_T;
// extern KungFu_Datatype KungFu_UINT8_T;
// extern KungFu_Datatype KungFu_INT16_T;
// extern KungFu_Datatype KungFu_UINT16_T;
// extern KungFu_Datatype KungFu_INT32_T;
// extern KungFu_Datatype KungFu_UINT32_T;
// extern KungFu_Datatype KungFu_INT64_T;
// extern KungFu_Datatype KungFu_UINT64_T;
extern KungFu_Datatype KungFu_FLOAT;
extern KungFu_Datatype KungFu_DOUBLE;
// extern KungFu_Datatype KungFu_LONG_DOUBLE;

typedef int KungFu_Op;

extern KungFu_Op KungFu_MAX;
extern KungFu_Op KungFu_MIN;
extern KungFu_Op KungFu_SUM;

#ifdef __cplusplus
}
#endif
