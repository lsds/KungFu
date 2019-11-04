package store

import (
	"errors"
	"sync"
)

var errSizeNotMatch = errors.New("size not match")

type Blob struct {
	sync.RWMutex

	Data []byte
}

func NewBlob(size int) *Blob {
	return &Blob{Data: make([]byte, size)}
}

func (b *Blob) CopyFrom(buf []byte) error {
	if len(b.Data) != len(buf) {
		return errSizeNotMatch
	}
	copy(b.Data, buf)
	return nil
}
