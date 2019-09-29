package rchannel

import (
	"sync"
)

// Reuse pool: chunk size -> pool.
var buffers = map[int]*sync.Pool{}
var buffersMux sync.Mutex

const pooledSize int = 512 // Minimum chunk size that is reused, reusing chunks too small will result in overhead.

// PutBuf puts a chunk to reuse pool if it can be reused.
func PutBuf(buf []byte) {
	size := cap(buf)
	if size < pooledSize {
		return
	}
	buffersMux.Lock()
	defer buffersMux.Unlock()
	if c := buffers[size]; c != nil {
		c.Put(buf[:0])
	}
}

// GetBuf gets a chunk from reuse pool or creates a new one if reuse failed.
func GetBuf(size int) []byte {
	if size < pooledSize {
		return make([]byte, 0, size)
	}

	buffersMux.Lock()
	c := buffers[size]
	if c == nil {
		c = new(sync.Pool)
		buffers[size] = c
	}
	buffersMux.Unlock()

	v := c.Get()
	if v != nil {
		return v.([]byte)
	}

	return make([]byte, 0, size)
}