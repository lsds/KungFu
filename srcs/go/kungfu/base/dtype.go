package base

// #include "kungfu/dtype.h"
import "C"

type DataType C.KungFu_Datatype

const (
	U8  DataType = C.KungFu_UINT8
	U16 DataType = C.KungFu_UINT16
	U32 DataType = C.KungFu_UINT32
	U64 DataType = C.KungFu_UINT64

	I8  DataType = C.KungFu_INT8
	I16 DataType = C.KungFu_INT16
	I32 DataType = C.KungFu_INT32
	I64 DataType = C.KungFu_INT64

	F16 DataType = C.KungFu_FLOAT16
	F32 DataType = C.KungFu_FLOAT
	F64 DataType = C.KungFu_DOUBLE
)

func (t DataType) Size() int {
	return int(C.kungfu_type_size(C.KungFu_Datatype(t)))
}

var dtypeNames = map[DataType]string{
	U8:  "u8",
	U16: "u16",
	U32: "u32",
	U64: "u64",

	I8:  "i8",
	I16: "i16",
	I32: "i32",
	I64: "i64",

	F16: "f16",
	F32: "f32",
	F64: "f64",
}

func (t DataType) String() string {
	return dtypeNames[t]
}
