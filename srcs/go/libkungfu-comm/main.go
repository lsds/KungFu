package main

import (
	"os"
	"reflect"
	"unsafe"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

// #include <kungfu.h>
// #include <kungfu_base.h>
import "C"

var kungfu *kf.Kungfu

//export GoKungfuInit
func GoKungfuInit(algo C.KungFu_AllReduceAlgo) int {
	var err error
	config := kf.Config{Algo: kb.KungFu_AllReduceAlgo(algo)}
	kungfu, err = kf.New(config)
	if err != nil {
		utils.ExitErr(err)
	}
	return kungfu.Start()
}

//export GoKungfuFinalize
func GoKungfuFinalize() int {
	return kungfu.Close()
}

//export GoKungfuAllReduce
func GoKungfuAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toBuffer(sendBuf, count, dtype),
		RecvBuf: toBuffer(recvBuf, count, dtype),
		OP:      kb.KungFu_Op(op),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		return sess.AllReduce(w)
	}
	go func() {
		sess.AllReduce(w)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuReduce
func GoKungfuReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toBuffer(sendBuf, count, dtype),
		RecvBuf: toBuffer(recvBuf, count, dtype),
		OP:      kb.KungFu_Op(op),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		return sess.Reduce(w)
	}
	go func() {
		sess.Reduce(w)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuBroadcast
func GoKungfuBroadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toBuffer(sendBuf, count, dtype),
		RecvBuf: toBuffer(recvBuf, count, dtype),
		// OP:      0, // FIXME: assert that OP is not used
		Name: C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		return sess.Broadcast(w)
	}
	go func() {
		sess.Broadcast(w)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuGetAlgoFromEnv
func GoKungfuGetAlgoFromEnv() C.KungFu_AllReduceAlgo {
	name := os.Getenv(kb.AllReduceAlgoEnvKey)
	return C.KungFu_AllReduceAlgo(kb.ParseAlgo(name))
}

func main() {}

func toBuffer(ptr unsafe.Pointer, count int, dtype C.KungFu_Datatype) *kb.Buffer {
	dt := kb.KungFu_Datatype(dtype)
	size := count * dt.Size()
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  size,
		Cap:  size,
	}
	return &kb.Buffer{
		Data:  *(*[]byte)(unsafe.Pointer(sh)),
		Count: count,
		Type:  dt,
	}
}
