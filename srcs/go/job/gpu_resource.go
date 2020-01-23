package job

import "sync"

type GPUPool struct {
	sync.Mutex
	cap int
	ids []int
}

func NewGPUPool(n int) *GPUPool {
	var ids []int
	for i := 0; i < n; i++ {
		ids = append(ids, i)
	}
	return &GPUPool{cap: n, ids: ids}
}

func (p *GPUPool) Get() int {
	p.Lock()
	defer p.Unlock()
	if len(p.ids) <= 0 {
		return -1
	}
	id := p.ids[0]
	p.ids = p.ids[1:]
	return id
}

func (p *GPUPool) Put(id int) {
	p.Lock()
	defer p.Unlock()
	if 0 <= id && id < p.cap {
		p.ids = append(p.ids, id)
	}
}
