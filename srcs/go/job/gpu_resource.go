package job

import (
	"errors"
	"sync"

	"github.com/lsds/KungFu/srcs/go/utils"
)

// GPUPool manages GPU ids
type GPUPool struct {
	sync.Mutex
	cap  int
	mask []bool
}

// NewGPUPool create a new GPUPool of given size
func NewGPUPool(n int) *GPUPool {
	var mask []bool
	for i := 0; i < n; i++ {
		mask = append(mask, true)
	}
	return &GPUPool{cap: n, mask: mask}
}

// Get returns the smallest GPU id that is available
func (p *GPUPool) Get() int {
	p.Lock()
	defer p.Unlock()
	for i := range p.mask {
		if p.mask[i] {
			p.mask[i] = false
			return i
		}
	}
	return -1
}

var errGPUNotAllocated = errors.New("GPU not allocated")

// Put puts an GPU id back to the pool
func (p *GPUPool) Put(id int) {
	p.Lock()
	defer p.Unlock()
	if 0 <= id && id < p.cap {
		if p.mask[id] {
			utils.ExitErr(errGPUNotAllocated)
		}
		p.mask[id] = true
	}
}
