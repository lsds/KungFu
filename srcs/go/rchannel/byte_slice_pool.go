package rchannel

import (
	"sync"
)

// ByteSlicePool reuse byte slices: chunk size -> pool.
type ByteSlicePool struct {
	sync.Mutex
	buffers map[uint32]*sync.Pool
}

const minBufSize uint32 = 512 // Minimum chunk size that is reused, reusing chunks too small will result in overhead.

var bsPool = ByteSlicePool{
	buffers: map[uint32]*sync.Pool{},
}

// PutBuf puts a chunk to reuse pool if it can be reused.
func PutBuf(buf []byte) {
	size := uint32(cap(buf))
	if size < minBufSize {
		return
	}
	bsPool.Lock()
	defer bsPool.Unlock()
	if c := bsPool.buffers[size]; c != nil {
		c.Put(buf)
	}
}

// GetBuf gets a chunk from reuse pool or creates a new one if reuse failed.
func GetBuf(size uint32) []byte {
	if size < minBufSize {
		return make([]byte, size)
	}

	bsPool.Lock()
	c := bsPool.buffers[size]
	if c == nil {
		c = new(sync.Pool)
		bsPool.buffers[size] = c
	}
	bsPool.Unlock()

	v := c.Get()
	if v != nil {
		return v.([]byte)
	}

	return make([]byte, size)
}
