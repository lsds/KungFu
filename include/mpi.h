#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int MPI_Datatype;
typedef int MPI_Op;
typedef struct gompi_communicator_t *MPI_Comm;

extern MPI_Datatype MPI_INT;
extern MPI_Datatype MPI_INT8_T;
extern MPI_Datatype MPI_UINT8_T;
extern MPI_Datatype MPI_INT16_T;
extern MPI_Datatype MPI_UINT16_T;
extern MPI_Datatype MPI_INT32_T;
extern MPI_Datatype MPI_UINT32_T;
extern MPI_Datatype MPI_INT64_T;
extern MPI_Datatype MPI_UINT64_T;
extern MPI_Datatype MPI_FLOAT;
extern MPI_Datatype MPI_DOUBLE;
extern MPI_Datatype MPI_LONG_DOUBLE;

extern MPI_Op MPI_MAX;
extern MPI_Op MPI_MIN;
extern MPI_Op MPI_SUM;

#ifdef __cplusplus
}
#endif
