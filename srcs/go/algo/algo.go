package algo

// #include <algo.h>
import "C"
import (
	"unsafe"

	"github.com/luomai/kungfu/srcs/go/wire"
)

// AddBy performs ys[i] += xs[i] for vectors ys and xs
func AddBy(ys []byte, xs []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op) {
	C.std_transform_2(ptr(xs), ptr(ys), ptr(ys), C.int(count), C.int(dtype), C.int(op))
}

func ptr(bs []byte) unsafe.Pointer {
	return unsafe.Pointer(&bs[0])
}
