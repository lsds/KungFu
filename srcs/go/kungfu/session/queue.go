package session

import (
	"errors"
	"strconv"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/rchannel/handler"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

var (
	queueID   map[uint64]int
	queueIDMu sync.Mutex
)

func init() {
	queueID = make(map[uint64]int)
}

func getQueueID(src, dst int) int {
	queueIDMu.Lock()
	defer queueIDMu.Unlock()
	key := uint64(src)<<32 | uint64(dst)
	id := queueID[key]
	queueID[key]++
	return id
}

type Queue struct {
	ID       int
	name     string
	self     int
	src, dst int
	srcID    plan.PeerID
	dstID    plan.PeerID
	client   *client.Client
	handler  *handler.QueueHandler
}

func (q *Queue) Get() ([]byte, error) {
	assert.True(q.self == q.dst)
	log.Errorf("getting from %s %s", q.dstID, q.name)
	m := q.handler.Get(q.srcID, q.name)
	return m.Data, nil
}

func (q *Queue) Put(bs []byte) error {
	assert.True(q.self == q.src)
	if err := q.client.Send(q.dstID.WithName(q.name), bs, connection.ConnQueue, connection.NoFlag); err != nil {
		return err
	}
	// TODO: wait ACK
	return nil
}

var (
	errInvalidQueueEndpoints = errors.New("invalid queue endpoints")
)

func (sess *Session) NewQueue(src, dst int) (*Queue, error) {
	if src != sess.rank && dst != sess.rank {
		return nil, errInvalidQueueEndpoints
	}
	id := getQueueID(src, dst)
	// TODO: handle shake by performing subset consistency check
	q := &Queue{
		ID:      id,
		name:    strconv.Itoa(id),
		self:    sess.rank,
		src:     src,
		dst:     dst,
		srcID:   sess.peers[src],
		dstID:   sess.peers[dst],
		client:  sess.client,
		handler: sess.queueHandler,
	}
	// TODO: handle shake
	return q, nil
}

type QueuePair struct {
	sendQ *Queue
	recvQ *Queue
}

func (sess *Session) NewQueuePair(src, dst int) (*QueuePair, error) {
	sendQ, err := sess.NewQueue(src, dst)
	if err != nil {
		return nil, err
	}
	recvQ, err := sess.NewQueue(dst, src)
	if err != nil {
		return nil, err
	}
	return &QueuePair{
		sendQ: sendQ,
		recvQ: recvQ,
	}, nil
}

func (qp *QueuePair) Send(x []byte) error {
	return qp.sendQ.Put(x)
}

func (qp *QueuePair) Recv() ([]byte, error) {
	return qp.recvQ.Get()
}
