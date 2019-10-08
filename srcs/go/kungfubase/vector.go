package kungfubase

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/utils"
)

type Buffer struct {
	Data  []byte
	Count int
	Type  KungFu_Datatype
}

func NewBuffer(count int, dtype KungFu_Datatype) *Buffer {
	return &Buffer{
		Data:  make([]byte, count*dtype.Size()),
		Count: count,
		Type:  dtype,
	}
}

// Slice returns a new Buffer that points to a subset of the original Buffer.
// 0 <= begin < end <= count - 1
func (b *Buffer) Slice(begin, end int) *Buffer {
	return &Buffer{
		Data:  b.Data[begin*b.Type.Size() : end*b.Type.Size()],
		Count: end - begin,
		Type:  b.Type,
	}
}

func (b *Buffer) CopyFrom(c *Buffer) {
	if err := b.copyFrom(c); err != nil {
		utils.ExitErr(err)
	}
}

func (b *Buffer) MaybeCopyFrom(c *Buffer) error {
	return b.copyFrom(c)
}

func (b *Buffer) copyFrom(c *Buffer) error {
	if b.Count != c.Count {
		return fmt.Errorf("Buffer::Copy error: inconsistent count: %d vs %d", b.Count, c.Count)
	}
	if b.Type != c.Type {
		return fmt.Errorf("Buffer::Copy error: inconsistent type: %s vs %s", b.Type, c.Type)
	}
	copy(b.Data, c.Data)
	return nil
}

func (b *Buffer) AsF32() []float32 {
	if b.Type != KungFu_FLOAT {
		utils.ExitErr(fmt.Errorf("buffer type is %d", b.Type))
	}
	sh := &reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&b.Data[0])),
		Len:  b.Count,
		Cap:  b.Count,
	}
	return *(*[]float32)(unsafe.Pointer(sh))
}
