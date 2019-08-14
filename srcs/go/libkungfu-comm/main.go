package main

import (
	"fmt"
	"os"
	"reflect"
	"unsafe"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
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

//export GoKungfuClusterSize
func GoKungfuClusterSize(version int) int {
	if version < 0 {
		sess := kungfu.CurrentSession()
		return sess.ClusterSize()
	}
	log.Warnf("GoKungfuClusterSize for version >= 0 is NOT supported, using current version")
	sess := kungfu.CurrentSession()
	return sess.ClusterSize()
}

//export GoKungfuRank
func GoKungfuRank(version int) int {
	if version < 0 {
		sess := kungfu.CurrentSession()
		return sess.Rank()
	}
	log.Warnf("GoKungfuClusterSize for version >= 0 is NOT supported, using current version")
	sess := kungfu.CurrentSession()
	return sess.Rank()
}

//export GoKungfuBarrier
func GoKungfuBarrier(done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	if done == nil {
		return sess.Barrier()
	}
	go func() {
		sess.Barrier()
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuRequest
func GoKungfuRequest(rank int, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	goName := C.GoString(name) // copy *C.char into go string before entering goroutine
	b := toBuffer(buf, count, dtype)
	if done == nil {
		// Synchronous case
		return sess.Request(rank, goName, b)
	}
	go func() {
		sess.Request(rank, goName, b)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuSave
func GoKungfuSave(name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	goName := C.GoString(name) // copy *C.char into go string before entering goroutine
	b := toBuffer(buf, count, dtype)
	if done == nil {
		return sess.Save(goName, b)
	}
	go func() {
		sess.Save(goName, b)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
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

//export GoKungfuGather
func GoKungfuGather(sendBuf unsafe.Pointer, sendCount int, sendDtype C.KungFu_Datatype,
	recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype,
	name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toBuffer(sendBuf, sendCount, sendDtype),
		RecvBuf: toBuffer(recvBuf, recvCount, recvDtype),
		// OP:      0, // FIXME: assert that OP is not used
		Name: C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		return sess.Gather(w)
	}
	go func() {
		sess.Gather(w)
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuGetPeerLatencies
func GoKungfuGetPeerLatencies(recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype) int {
	results := toBuffer(recvBuf, recvCount, recvDtype).AsF32()
	sess := kungfu.CurrentSession()
	latencies := sess.GetPeerLatencies()
	// FIXME: check length
	for i := range results {
		results[i] = float32(latencies[i])
	}
	return 0
}

//export GoKungfuProposeUpdate
func GoKungfuProposeUpdate(token *C.char, newSize int, accepted *C.char) int {
	ok := true // FIXME: compute ok by all reduce
	*accepted = boolToChar(ok)
	if ok {
		kungfu.ProposeUpdate(C.GoString(token), newSize)
	}
	return 0
}

//export GoKungfuUpdateCluster
func GoKungfuUpdateCluster(token *C.char, exist *C.char) int {
	*exist = boolToChar(kungfu.UpdateSession(C.GoString(token)))
	return 0
}

//export GoKungfuGetAlgoFromEnv
func GoKungfuGetAlgoFromEnv() C.KungFu_AllReduceAlgo {
	name := os.Getenv(kb.AllReduceAlgoEnvKey)
	return C.KungFu_AllReduceAlgo(kb.ParseAlgo(name))
}

func main() {}

func toBuffer(ptr unsafe.Pointer, count int, dtype C.KungFu_Datatype) *kb.Buffer {
	if ptr == nil {
		if count > 0 {
			utils.ExitErr(fmt.Errorf("toBuffer: ptr is nil but count = %d", count))
		}
	}
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

func boolToChar(v bool) C.char {
	if v {
		return C.char(1)
	}
	return C.char(0)
}
