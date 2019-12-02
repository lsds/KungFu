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

//export GoKungfuAllReduce
func GoKungfuAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("AllReduce", sess.AllReduce, w, done)
}

//export GoKungfuFPGAAllReduce
func GoKungfuFPGAAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char) int {
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	//call a function here that is blocking
	return 0;
//	return callCollectiveOP("FPGAAllReduce", sess.FPGAAllReduce, w, done)
}

//export GoKungfuReduce
func GoKungfuReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Reduce", sess.Reduce, w, done)
}

//export GoKungfuBroadcast
func GoKungfuBroadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Broadcast", sess.Broadcast, w, done)
}

//export GoKungfuGather
func GoKungfuGather(sendBuf unsafe.Pointer, sendCount int, sendDtype C.KungFu_Datatype,
	recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype,
	name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, sendCount, sendDtype),
		RecvBuf: toVector(recvBuf, recvCount, recvDtype),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	return callCollectiveOP("Gather", sess.Gather, w, done)
}

func callCollectiveOP(name string, op func(kf.Workspace) error, w kf.Workspace, done *C.callback_t) int {
	return callOP(name, func() error { return op(w) }, done)
}
