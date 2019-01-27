package kungfubase

import "unsafe"

// #include <kungfu_base.h>
import "C"

// Transform performs ys[i] += xs[i] for vectors ys and xs
func Transform(ys []byte, xs []byte, count int, dtype KungFu_Datatype, op KungFu_Op) {
	C.std_transform_2(ptr(xs), ptr(ys), ptr(ys), C.int(count), C.int(dtype), C.int(op))
}

// Transform2 performs zs[i] = xs[i] + ys[i] for vectors ys and xs
func Transform2(zs, xs, ys []byte, count int, dtype KungFu_Datatype, op KungFu_Op) {
	C.std_transform_2(ptr(xs), ptr(ys), ptr(zs), C.int(count), C.int(dtype), C.int(op))
}

func ptr(bs []byte) unsafe.Pointer {
	return unsafe.Pointer(&bs[0])
}
