package rchannel

import (
	"reflect"
	"sync"
	"unsafe"
)

// #include "pool.h"
import "C"

type pool struct {
	mu sync.Mutex
	p  *C.pool_t
}

var (
	defaultPool = newPool()

	alloc = defaultPool.get
	free  = defaultPool.put

	Alloc = alloc
	Free  = free
)

func newPool() *pool {
	return &pool{
		p: C.new_pool(),
	}
}

func (p *pool) get(n int) []byte {
	p.mu.Lock()
	defer p.mu.Unlock()
	ptr := unsafe.Pointer(C.get_buffer(p.p, C.int(n)))
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  n,
		Cap:  n,
	}
	return *(*[]byte)(unsafe.Pointer(sh))
}

func (p *pool) put(bs []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()
	C.put_buffer(p.p, unsafe.Pointer(&bs[0]))
}
