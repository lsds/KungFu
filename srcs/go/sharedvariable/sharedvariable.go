package sharedvariable

import (
	"io"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type SharedVariable struct {
	sync.RWMutex

	data *kb.Buffer
}

func NewSharedVariable(count int, dtype kb.KungFu_Datatype) *SharedVariable {
	return &SharedVariable{
		data: kb.NewBuffer(count, dtype),
	}
}

func (v *SharedVariable) Get(buf *kb.Buffer) error {
	v.RLock()
	defer v.RUnlock()
	return buf.MaybeCopyFrom(v.data)
}

func (v *SharedVariable) Put(buf *kb.Buffer) error {
	v.Lock()
	defer v.Unlock()
	return v.data.MaybeCopyFrom(buf)
}

func (v *SharedVariable) Add(buf *kb.Buffer, output *kb.Buffer) error {
	if !v.data.SameType(buf) {
		return errConflict
	}
	if output != nil && !v.data.SameType(output) {
		return errConflict
	}
	v.Lock()
	defer v.Unlock()
	kb.Transform(v.data, buf, kb.KungFu_SUM)
	if output != nil {
		output.CopyFrom(v.data)
	}
	return nil
}

func (v *SharedVariable) ReadFrom(r io.Reader) error {
	v.Lock()
	defer v.Unlock()
	return readBuf(r, v.data)
}

// f must be readonly
func (v *SharedVariable) do(f func(*kb.Buffer)) error {
	v.RLock()
	defer v.RUnlock()
	f(v.data)
	return nil
}
