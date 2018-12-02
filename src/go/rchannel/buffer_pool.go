package rchannel

import (
	"sync"
)

type BufferPool struct {
	sync.Mutex
	buffers map[Addr]chan *Message
}

func newBufferPool() *BufferPool {
	return &BufferPool{
		buffers: make(map[Addr]chan *Message),
	}
}

func (p *BufferPool) require(a Addr) chan *Message {
	p.Lock()
	defer p.Unlock()
	m, ok := p.buffers[a]
	if !ok {
		m = make(chan *Message, 10)
		p.buffers[a] = m
	}
	return m
}
