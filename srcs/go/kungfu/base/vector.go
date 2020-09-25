package base

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

type Vector struct {
	Data  []byte
	Count int
	Type  DataType
}

func NewVector(count int, dtype DataType) *Vector {
	return &Vector{
		Data:  make([]byte, count*dtype.Size()),
		Count: count,
		Type:  dtype,
	}
}

// Slice returns a new Vector that points to a subset of the original Vector.
// 0 <= begin < end <= count - 1
func (b *Vector) Slice(begin, end int) *Vector {
	return &Vector{
		Data:  b.Data[begin*b.Type.Size() : end*b.Type.Size()],
		Count: end - begin,
		Type:  b.Type,
	}
}

func (b *Vector) CopyFrom(c *Vector) {
	assert.OK(b.copyFrom(c))
}

func (b *Vector) copyFrom(c *Vector) error {
	if b.Count != c.Count {
		return fmt.Errorf("Vector::Copy error: inconsistent count: %d vs %d", b.Count, c.Count)
	}
	if b.Type != c.Type {
		return fmt.Errorf("Vector::Copy error: inconsistent type: %d vs %d", b.Type, c.Type)
	}
	copy(b.Data, c.Data)
	return nil
}

func (b *Vector) sliceHeader() unsafe.Pointer {
	sh := &reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&b.Data[0])),
		Len:  b.Count,
		Cap:  b.Count,
	}
	return unsafe.Pointer(sh)
}

func (b *Vector) AsF32() []float32 {
	assert.True(b.Type == F32)
	return *(*[]float32)(b.sliceHeader())
}

func (b *Vector) AsF64() []float64 {
	assert.True(b.Type == F64)
	return *(*[]float64)(b.sliceHeader())
}

func (b *Vector) AsI8() []int8 {
	assert.True(b.Type == I8)
	return *(*[]int8)(b.sliceHeader())
}

func (b *Vector) AsI32() []int32 {
	assert.True(b.Type == I32)
	return *(*[]int32)(b.sliceHeader())
}

func (b *Vector) AsI64() []int64 {
	assert.True(b.Type == I64)
	return *(*[]int64)(b.sliceHeader())
}
