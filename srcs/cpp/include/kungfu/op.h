#pragma once
#include <kungfu/dtype.h>

#ifdef __cplusplus
extern "C" {
#endif

enum KungFu_Op {
    KungFu_SUM,
    KungFu_MIN,
    KungFu_MAX,
    KungFu_PROD,
};

typedef enum KungFu_Op KungFu_Op;

extern void std_transform_2(const void *input1, const void *input2,
                            void *output, const int n, const KungFu_Datatype dt,
                            const KungFu_Op o);

#ifdef __cplusplus
}
#endif
