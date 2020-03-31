package handler

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
)

type BufferPool struct {
	sync.Mutex
	qSize   int
	buffers map[plan.Addr]chan *connection.Message
}

func newBufferPool(qSize int) *BufferPool {
	return &BufferPool{
		qSize:   qSize,
		buffers: make(map[plan.Addr]chan *connection.Message),
	}
}

func (p *BufferPool) require(a plan.Addr) chan *connection.Message {
	p.Lock()
	defer p.Unlock()
	m, ok := p.buffers[a]
	if !ok {
		m = make(chan *connection.Message, p.qSize)
		p.buffers[a] = m
	}
	return m
}
