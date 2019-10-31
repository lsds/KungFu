package rchannel

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type BufferPool struct {
	sync.Mutex
	qSize   int
	buffers map[plan.Addr]chan *Message
}

func newBufferPool(qSize int) *BufferPool {
	return &BufferPool{
		qSize:   qSize,
		buffers: make(map[plan.Addr]chan *Message),
	}
}

func (p *BufferPool) require(a plan.Addr) chan *Message {
	p.Lock()
	defer p.Unlock()
	m, ok := p.buffers[a]
	if !ok {
		m = make(chan *Message, p.qSize)
		p.buffers[a] = m
	}
	return m
}
