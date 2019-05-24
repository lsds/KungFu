package main

import (
	"os"
	"reflect"
	"unsafe"
	"fmt"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
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

//export GoKungfuClusterSize
func GoKungfuClusterSize() int {
	sess := kungfu.CurrentSession()
	return sess.ClusterSize()
}

//export GoKungfuRank
func GoKungfuRank() int {
	sess := kungfu.CurrentSession()
	return sess.Rank()
}

//export GoKungfuRegisterDataCallback
func GoKungfuRegisterDataCallback(name *C.char, handle *C.data_callback_t) int {
	sess := kungfu.CurrentSession()
	return sess.RegisterDataCallback(C.GoString(name), func(msg *rch.Message) {
		C.invoke_data_callback(handle, unsafe.Pointer(&msg.Data[0]), C.int(msg.Length))
	})
}

//export GoKungfuUnregisterDataCallback
func GoKungfuUnregisterDataCallback(name *C.char) int {
	sess := kungfu.CurrentSession()
	return sess.UnregisterDataCallback(C.GoString(name))
}

//export GoKungfuSendTo
func GoKungfuSendTo(rank int, sendBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toBuffer(sendBuf, count, dtype),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()

	if done == nil {
		// Synchronous case
		return sess.SendTo(rank, w)
	}
	go func() {
		// Asynchronous case
		sess.SendTo(rank, w)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuRequestVar
func GoKungfuRequestVar(rank int, name *C.char, count int, dtype C.KungFu_Datatype, recvBuf unsafe.Pointer, done *C.callback_t) int {
	fmt.Printf("%s", "Inside go")

	w := kf.Workspace{
		RecvBuf: toBuffer(recvBuf, count, dtype),
		Name:    C.GoString(name),
	}
	sess := kungfu.CurrentSession()

	if done == nil {
		// Synchronous case
		return sess.SendTo(rank, w)
	}

	return 0
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
