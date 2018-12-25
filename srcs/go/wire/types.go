package wire

// #include <kungfu.h>
import "C"
import "strconv"

type KungFu_Datatype C.KungFu_Datatype

var (
	KungFu_INT    = KungFu_Datatype(C.KungFu_INT)
	KungFu_FLOAT  = KungFu_Datatype(C.KungFu_FLOAT)
	KungFu_DOUBLE = KungFu_Datatype(C.KungFu_DOUBLE)
)

func (dtype KungFu_Datatype) Size() int {
	switch dtype {
	case KungFu_INT:
		return 4
	case KungFu_FLOAT:
		return 4
	default:
		panic("unknown dtype: " + strconv.Itoa(int(dtype)))
	}
}

type KungFu_Op C.KungFu_Op

var (
	KungFu_SUM KungFu_Op = KungFu_Op(C.KungFu_SUM)
)
