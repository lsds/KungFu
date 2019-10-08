#pragma once
#include "dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    sum,
    min,
    max,
    prod,
} op;

extern void std_transform_2(const void *input1, const void *input2,
                            void *output, const int n, const dtype dt,
                            const op o);

#ifdef __cplusplus
}
#endif
