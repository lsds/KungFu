package base

import "unsafe"

// #cgo CXXFLAGS: -std=c++11
// #include "kungfu/op.h"
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
	Transform2(y, x, y, op)
}

// Transform2 performs z[i] = x[i] + y[i] for vectors z and x, y.
func Transform2(z, x, y *Vector, op OP) {
	// Assuming Count and Type are consistent
	C.std_transform_2(
		// ptr(x.Data), // panic when x.Data is returned from bytes.Buffer
		// ptr(y.Data),
		// ptr(z.Data),
		// https://github.com/lsds/KungFu/issues/149
		unsafe.Pointer(&x.Data[0]),
		unsafe.Pointer(&y.Data[0]),
		unsafe.Pointer(&z.Data[0]),
		C.int(z.Count), C.KungFu_Datatype(z.Type), C.KungFu_Op(op))
}

// panic: runtime error: cgo argument has Go pointer to Go pointer
// func ptr(bs []byte) unsafe.Pointer {
// 	return unsafe.Pointer(&bs[0])
// }
