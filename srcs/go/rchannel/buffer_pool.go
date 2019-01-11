package rchannel

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type BufferPool struct {
	sync.Mutex
	buffers map[plan.Addr]chan *Message
}

func newBufferPool() *BufferPool {
	return &BufferPool{
		buffers: make(map[plan.Addr]chan *Message),
	}
}

func (p *BufferPool) require(a plan.Addr) chan *Message {
	p.Lock()
	defer p.Unlock()
	m, ok := p.buffers[a]
	if !ok {
		m = make(chan *Message, 10)
		p.buffers[a] = m
	}
	return m
}
