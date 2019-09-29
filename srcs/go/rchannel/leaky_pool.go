package rchannel

import (
	"sync"
)

// Reuse pool: chunk size -> pool.
var buffers = map[uint32]*sync.Pool{}
var mu sync.Mutex

const minBufSize uint32 = 512 // Minimum chunk size that is reused, reusing chunks too small will result in overhead.

// PutBuf puts a chunk to reuse pool if it can be reused.
func PutBuf(buf []byte) {
	size := uint32(cap(buf))
	if size < minBufSize {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	if c := buffers[size]; c != nil {
		c.Put(buf)
	}
}

// GetBuf gets a chunk from reuse pool or creates a new one if reuse failed.
func GetBuf(size uint32) []byte {
	if size < minBufSize {
		return make([]byte, size)
	}

	mu.Lock()
	c := buffers[size]
	if c == nil {
		c = new(sync.Pool)
		buffers[size] = c
	}
	mu.Unlock()

	v := c.Get()
	if v != nil {
		return v.([]byte)
	}

	return make([]byte, size)
}
