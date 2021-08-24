package main

/*
#include <kungfu/dtype.h>
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/kungfu/session"
)

type queueUID struct {
	src, dst, id int
}

var (
	queues map[queueUID]*session.Queue
)

func init() {
	queues = make(map[queueUID]*session.Queue)
}

//export GoKungfuQueueNew
func GoKungfuQueueNew(src, dst int, queueID *C.int) int {
	sess := defaultPeer.CurrentSession()
	q, err := sess.NewQueue(src, dst)
	if err != nil {
		return errorCode(`GoKungfuQueueNew`, err)
	}
	uid := queueUID{src: src, dst: dst, id: q.ID}
	queues[uid] = q
	*queueID = C.int(q.ID)
	return 0
}

var errQueueNotCreated = errors.New(`queue not created`)

//export GoKungfuQueueGet
func GoKungfuQueueGet(src, dst int, queueID C.int, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype) int {
	uid := queueUID{src: src, dst: dst, id: int(queueID)}
	q, ok := queues[uid]
	if !ok {
		return errorCode(`GoKungfuQueueGet`, errQueueNotCreated)
	}
	data, err := q.Get()
	if err != nil {
		return errorCode(`GoKungfuQueueGet`, err)
	}
	x := toVector(buf, count, dtype)
	if len(x.Data) != len(data) {
		return errorCode(`GoKungfuQueueGet`, err)
	}
	copy(x.Data, data)
	return 0
}

//export GoKungfuQueuePut
func GoKungfuQueuePut(src, dst int, queueID C.int, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype) int {
	uid := queueUID{src: src, dst: dst, id: int(queueID)}
	q, ok := queues[uid]
	if !ok {
		return errorCode(`GoKungfuQueueGet`, errQueueNotCreated)
	}
	x := toVector(buf, count, dtype)
	q.Put(x.Data)
	return 0
}
