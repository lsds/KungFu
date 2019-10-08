package kungfubase

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/utils"
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
	if err := b.copyFrom(c); err != nil {
		utils.ExitErr(err)
	}
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

func (b *Vector) AsF32() []float32 {
	if b.Type != F32 {
		utils.ExitErr(fmt.Errorf("Vector type is %d", b.Type))
	}
	sh := &reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&b.Data[0])),
		Len:  b.Count,
		Cap:  b.Count,
	}
	return *(*[]float32)(unsafe.Pointer(sh))
}
