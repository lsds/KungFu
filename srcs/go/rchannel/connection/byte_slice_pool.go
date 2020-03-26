package connection

import (
	"sync"
)

// ByteSlicePool reuse byte slices: chunk size -> pool.
type ByteSlicePool struct {
	sync.Mutex
	buffers map[uint32]*sync.Pool
}

const minBufSize uint32 = 512 // Minimum chunk size that is reused, reusing chunks too small will result in overhead.

var (
	defaultPool = newByteSlicePool()
	GetBuf      = defaultPool.GetBuf
	PutBuf      = defaultPool.PutBuf
)

// newByteSlicePool create a byte slice pool
func newByteSlicePool() *ByteSlicePool {
	return &ByteSlicePool{
		buffers: make(map[uint32]*sync.Pool),
	}
}

// PutBuf puts a chunk to reuse pool if it can be reused.
func (p *ByteSlicePool) PutBuf(buf []byte) {
	size := uint32(cap(buf))
	if size < minBufSize {
		return
	}
	p.Lock()
	defer p.Unlock()
	if c := p.buffers[size]; c != nil {
		c.Put(buf)
	}
}

// GetBuf gets a chunk from reuse pool or creates a new one if reuse failed.
func (p *ByteSlicePool) GetBuf(size uint32) []byte {
	if size < minBufSize {
		return make([]byte, size)
	}

	p.Lock()
	c, ok := p.buffers[size]
	if !ok {
		c = new(sync.Pool)
		p.buffers[size] = c
	}
	p.Unlock()

	if v := c.Get(); v != nil {
		return v.([]byte)
	}

	return make([]byte, size)
}
