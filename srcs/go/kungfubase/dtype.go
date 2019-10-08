package kungfubase

// #include "dtype.h"
import "C"

type DataType C.dtype

const (
	U8  DataType = C.u8
	U16 DataType = C.u16
	U32 DataType = C.u32
	U64 DataType = C.u64

	I8  DataType = C.i8
	I16 DataType = C.i16
	I32 DataType = C.i32
	I64 DataType = C.i64

	F16 DataType = C.f16
	F32 DataType = C.f32
	F64 DataType = C.f64
)

func (t DataType) Size() int {
	return int(C.dtype_size(C.dtype(t)))
}
