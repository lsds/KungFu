package wire

// #include <mpi.h>
import "C"
import "strconv"

type MPI_Datatype C.MPI_Datatype

var (
	MPI_INT    = MPI_Datatype(C.MPI_INT)
	MPI_FLOAT  = MPI_Datatype(C.MPI_FLOAT)
	MPI_DOUBLE = MPI_Datatype(C.MPI_DOUBLE)
)

func (dtype MPI_Datatype) Size() int {
	switch dtype {
	case MPI_INT:
		return 4
	case MPI_FLOAT:
		return 4
	default:
		panic("unknown dtype: " + strconv.Itoa(int(dtype)))
	}
}

type MPI_Op C.MPI_Op

var (
	MPI_SUM MPI_Op = MPI_Op(C.MPI_SUM)
)
