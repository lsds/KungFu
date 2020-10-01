package main

import (
	"unsafe"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
)

/*
#include <kungfu/callback.h>
#include <kungfu/dtype.h>
#include <kungfu/op.h>
*/
import "C"

//export GoKungfuBarrier
func GoKungfuBarrier(done *C.callback_t) int {
	sess := defaultPeer.CurrentSession()
	return callOP("Barrier", sess.Barrier, done)
}

//export GoKungfuConsensus
func GoKungfuConsensus(buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, pOK *C.char, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(buf, count, dtype),
		RecvBuf: toVector(unsafe.Pointer(pOK), 1, C.KungFu_Datatype(kb.I8)),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("Consensus", name, sess.Consensus, w, done)
}

//export GoKungfuAllReduce
func GoKungfuAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("AllReduce", name, sess.AllReduce, w, done)
}

//export GoKungfuCrossAllReduce
func GoKungfuCrossAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("CrossAllReduce", name, sess.CrossAllReduce, w, done)
}

//export GoKungfuMonitoredAllReduce
func GoKungfuMonitoredAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, pTree unsafe.Pointer /* TODO: return monitoring data */, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	np := sess.Size()

	var f func(w kb.Workspace) error
	if pTree == nil {
		f = func(w kb.Workspace) error { return sess.AllReduceWith(nil, w) }
	} else {
		tree := toVector(pTree, np, C.KungFu_INT32) // TODO: ensure pTree has size np in C++
		f = func(w kb.Workspace) error { return sess.AllReduceWith(tree.AsI32(), w) }
	}
	return callCollectiveOP("AllReduceWith", name, f, w, done)
}

//export GoKungfuAllGather
func GoKungfuAllGather(sendBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, recvBuf unsafe.Pointer, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	sess := defaultPeer.CurrentSession()
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count*sess.Size(), dtype),
		Name:    name,
	}
	return callCollectiveOP("AllGather", name, sess.AllGather, w, done)
}

//export GoKungfuReduce
func GoKungfuReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("Reduce", name, sess.Reduce, w, done)
}

//export GoKungfuBroadcast
func GoKungfuBroadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("Broadcast", name, sess.Broadcast, w, done)
}

//export GoKungfuGather
func GoKungfuGather(sendBuf unsafe.Pointer, sendCount int, sendDtype C.KungFu_Datatype, recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, sendCount, sendDtype),
		RecvBuf: toVector(recvBuf, recvCount, recvDtype),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("Gather", name, sess.Gather, w, done)
}

//export GoKungfuLocalReduce
func GoKungfuLocalReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("LocalReduce", name, sess.LocalReduce, w, done)
}

//export GoKungfuLocalBroadcast
func GoKungfuLocalBroadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kb.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		Name:    name,
	}
	sess := defaultPeer.CurrentSession()
	return callCollectiveOP("LocalBroadcast", name, sess.LocalBroadcast, w, done)
}

func callCollectiveOP(opName, name string, op func(kb.Workspace) error, w kb.Workspace, done *C.callback_t) int {
	return callOP(opName+"("+name+")", func() error { return op(w) }, done)
}
