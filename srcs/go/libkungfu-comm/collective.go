package main

import (
	"unsafe"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

/*
#include <kungfu/callback.h>
#include <kungfu/dtype.h>
#include <kungfu/op.h>
*/
import "C"

//export GoKungfuBarrier
func GoKungfuBarrier(done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	return callOP("Barrier", sess.Barrier, done)
}

//export GoKungfuConsensus
func GoKungfuConsensus(buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, pOK *C.char, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kf.Workspace{
		SendBuf: toVector(buf, count, dtype),
		RecvBuf: toVector(unsafe.Pointer(pOK), 1, C.KungFu_Datatype(kb.I8)),
		Name:    name,
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Consensus", name, sess.Consensus, w, done)
}

//export GoKungfuAllReduce
func GoKungfuAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("AllReduce", name, sess.AllReduce, w, done)
}

//export GoKungfuReduce
func GoKungfuReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Reduce", name, sess.Reduce, w, done)
}

//export GoKungfuBroadcast
func GoKungfuBroadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		Name:    name,
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Broadcast", name, sess.Broadcast, w, done)
}

//export GoKungfuGather
func GoKungfuGather(sendBuf unsafe.Pointer, sendCount int, sendDtype C.KungFu_Datatype, recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, sendCount, sendDtype),
		RecvBuf: toVector(recvBuf, recvCount, recvDtype),
		Name:    name,
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Gather", name, sess.Gather, w, done)
}

func callCollectiveOP(opName, name string, op func(kf.Workspace) error, w kf.Workspace, done *C.callback_t) int {
	return callOP(opName+"("+name+")", func() error { return op(w) }, done)
}
