package shmpool

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/utils/shm"
)

type Block struct {
	Offset int
	Size   int
}

type blockQueue chan Block

const defaultQueueCap = 1000

type Pool struct {
	smu sync.Mutex
	shm shm.Shm

	qmu             sync.Mutex
	top             int
	allocatedCounts map[int]int
	availableBlocks map[int]blockQueue
}

func New(name string) (*Pool, error) {
	shm, err := shm.New(name)
	if err != nil {
		return nil, err
	}
	return &Pool{
		shm:             shm,
		allocatedCounts: make(map[int]int),
		availableBlocks: make(map[int]blockQueue),
	}, nil
}

func (p *Pool) Close() {
	p.shm.Close()
}

func (p *Pool) ensureQueue(size int) {
	p.qmu.Lock()
	defer p.qmu.Unlock()
	if _, ok := p.availableBlocks[size]; !ok {
		p.availableBlocks[size] = make(blockQueue, defaultQueueCap)
	}
}

func (p *Pool) tryNewBlock(size int) {
	p.qmu.Lock()
	defer p.qmu.Unlock()
	if p.allocatedCounts[size] >= defaultQueueCap {
		return
	}
	b := Block{Offset: p.top, Size: size}
	p.top += size
	p.allocatedCounts[size]++
	p.availableBlocks[size] <- b
}

func (p *Pool) Get(size int) Block {
	p.ensureQueue(size)
	if len(p.availableBlocks[size]) == 0 {
		p.tryNewBlock(size)
	}
	b := <-p.availableBlocks[size]
	return b
}

func (p *Pool) Put(b Block) {
	p.qmu.Lock()
	defer p.qmu.Unlock()
	p.availableBlocks[b.Size] <- b
}

func (p *Pool) Write(b Block, bs []byte) {
	p.smu.Lock()
	defer p.smu.Unlock()
	p.shm.Seek(b.Offset)
	p.shm.Write(bs, len(bs))
}
