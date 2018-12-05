#include <mpi.h>
#include <mpi_types.hpp>

MPI_Datatype MPI_INT = mpi::type_encoder::value<int>();
MPI_Datatype MPI_FLOAT = mpi::type_encoder::value<float>();
MPI_Datatype MPI_DOUBLE = mpi::type_encoder::value<double>();

MPI_Op MPI_MAX = mpi::op_encoder::value<mpi::op_max>();
MPI_Op MPI_MIN = mpi::op_encoder::value<mpi::op_min>();
MPI_Op MPI_SUM = mpi::op_encoder::value<mpi::op_sum>();
