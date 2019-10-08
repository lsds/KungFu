package kungfubase

import "unsafe"

// #include "op.h"
import "C"

type OP C.KungFu_Op

const (
	SUM  OP = C.KungFu_SUM
	MIN  OP = C.KungFu_MIN
	MAX  OP = C.KungFu_MAX
	PROD OP = C.KungFu_PROD
)

// Transform performs y[i] += x[i] for vectors y and x
func Transform(y, x *Vector, op OP) {
	// Assuming Count and Type are consistent
	C.std_transform_2(ptr(x.Data), ptr(y.Data), ptr(y.Data), C.int(y.Count), C.KungFu_Datatype(y.Type), C.KungFu_Op(op))
}

// Transform2 performs z[i] = x[i] + y[i] for vectors z and x, y.
func Transform2(z, x, y *Vector, op OP) {
	// Assuming Count and Type are consistent
	C.std_transform_2(ptr(x.Data), ptr(y.Data), ptr(z.Data), C.int(z.Count), C.KungFu_Datatype(z.Type), C.KungFu_Op(op))
}

func ptr(bs []byte) unsafe.Pointer {
	return unsafe.Pointer(&bs[0])
}
