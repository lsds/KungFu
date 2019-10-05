package kungfubase

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/utils"
)

// #include <kungfu.h>
import "C"

const (
	CheckpointEnvKey = `KUNGFU_INIT_CKPT`
	ParentIDEnvKey   = `KUNGFU_PARENT_ID`
	HostListEnvKey   = `KUNGFU_HOST_LIST`

	PeerListEnvKey      = `KUNGFU_INIT_PEERS`
	HostSpecEnvKey      = `KUNGFU_HOST_SPEC`
	SelfSpecEnvKey      = `KUNGFU_SELF_SPEC` // self spec should never change during the life of a process
	InitSessEnvKey      = `KUNGFU_INIT_SESS`
	InitStepEnvKey      = `KUNGFU_INIT_STEP`
	AllReduceAlgoEnvKey = `KUNGFU_ALLREDUCE_ALGO` // FIXME: remove it
)

type KungFu_Datatype C.KungFu_Datatype

var (
	KungFu_UINT8  = KungFu_Datatype(C.KungFu_UINT8)
	KungFu_INT32  = KungFu_Datatype(C.KungFu_INT32)
	KungFu_INT64  = KungFu_Datatype(C.KungFu_INT64)
	KungFu_FLOAT  = KungFu_Datatype(C.KungFu_FLOAT)
	KungFu_DOUBLE = KungFu_Datatype(C.KungFu_DOUBLE)
)

func (dtype KungFu_Datatype) Size() int {
	return int(C.kungfu_type_size(C.KungFu_Datatype(dtype)))
}

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

type KungFu_Op C.KungFu_Op

var (
	KungFu_SUM = KungFu_Op(C.KungFu_SUM)
	KungFu_MIN = KungFu_Op(C.KungFu_MIN)
	KungFu_MAX = KungFu_Op(C.KungFu_MAX)
)

type KungFu_AllReduceAlgo C.KungFu_AllReduceAlgo

var (
	KungFu_Star   = KungFu_AllReduceAlgo(C.KungFu_StarAllReduce)
	KungFu_Ring   = KungFu_AllReduceAlgo(C.KungFu_RingAllReduce)
	KungFu_Clique = KungFu_AllReduceAlgo(C.KungFu_CliqueAllReduce)
	KungFu_Tree   = KungFu_AllReduceAlgo(C.KungFu_TreeAllReduce)

	algoNames = map[KungFu_AllReduceAlgo]string{
		KungFu_Star:   `STAR`,
		KungFu_Ring:   `RING`,
		KungFu_Clique: `CLIQUE`,
		KungFu_Tree:   `TREE`,
	}

	defaultAlgo = KungFu_Tree
)

func AllAlgoNames() []string {
	var names []string
	for _, name := range algoNames {
		names = append(names, name)
	}
	return names
}

func (a KungFu_AllReduceAlgo) String() string {
	for k, v := range algoNames {
		if a == k {
			return v
		}
	}
	return algoNames[defaultAlgo]
}

func ParseAlgo(s string) KungFu_AllReduceAlgo {
	for k, v := range algoNames {
		if s == v {
			return k
		}
	}
	return defaultAlgo
}
