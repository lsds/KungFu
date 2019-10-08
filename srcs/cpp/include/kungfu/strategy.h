#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum KungFu_AllReduceStrategy {
    KungFu_TreeAllReduce,
    KungFu_RingAllReduce,
    KungFu_StarAllReduce,
    KungFu_CliqueAllReduce,
};

typedef enum KungFu_AllReduceStrategy KungFu_AllReduceStrategy;

#ifdef __cplusplus
}
#endif
