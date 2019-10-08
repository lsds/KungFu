package kungfubase

import "unsafe"

// #include "op.h"
import "C"

type OP int

const (
	SUM  OP = C.sum
	MIN  OP = C.min
	MAX  OP = C.max
	PROD OP = C.prod
)

// Transform performs y[i] += x[i] for vectors y and x
func Transform(y, x *Buffer, op KungFu_Op) {
	// Assuming Count and Type are consistent
	C.std_transform_2(ptr(x.Data), ptr(y.Data), ptr(y.Data), C.int(y.Count), C.dtype(y.Type), C.op(op))
}

// Transform2 performs z[i] = x[i] + y[i] for vectors z and x, y.
func Transform2(z, x, y *Buffer, op KungFu_Op) {
	// Assuming Count and Type are consistent
	C.std_transform_2(ptr(x.Data), ptr(y.Data), ptr(z.Data), C.int(z.Count), C.dtype(z.Type), C.op(op))
}

func ptr(bs []byte) unsafe.Pointer {
	return unsafe.Pointer(&bs[0])
}
