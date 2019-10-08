package kungfubase

// #include "op.h"
import "C"

type OP int

const (
	SUM  OP = C.sum
	MIN  OP = C.min
	MAX  OP = C.max
	PROD OP = C.prod
)

// func transform(y, x *Vector, op OP) {
// 	transform2(y, x, y, op)
// }

// func transform2(z, x, y *Vector, op OP) {
// 	C.std_transform_2(ptr(x.Data), ptr(y.Data), ptr(z.Data), C.int(z.Count), C.dtype(z.Type), C.op(op))
// }

// func ptr(bs []byte) unsafe.Pointer {
// 	return unsafe.Pointer(&bs[0])
// }
