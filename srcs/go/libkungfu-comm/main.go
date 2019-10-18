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

/*
#include <kungfu/callback.h>
#include <kungfu/dtype.h>
#include <kungfu/op.h>
#include <kungfu/strategy.h>
*/
import "C"

var kungfu *kf.Kungfu

//export GoKungfuInit
func GoKungfuInit(strategy C.KungFu_AllReduceStrategy) int {
	var err error
	config := kf.Config{Strategy: kb.Strategy(strategy)}
	kungfu, err = kf.New(config)
	if err != nil {
		log.Errorf("failed to create KungFu instance: %v", err)
		return 1
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

//export GoKungfuBarrier
func GoKungfuBarrier(done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	if done == nil {
		return code("Barrier", sess.Barrier())
	}
	go func() {
		code("Barrier", sess.Barrier()) // FIXME: pass error code to done
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuRequest
func GoKungfuRequest(rank int, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	goName := C.GoString(name) // copy *C.char into go string before entering goroutine
	b := toVector(buf, count, dtype)
	if done == nil {
		// Synchronous case
		return code("Request", sess.Request(rank, goName, b))
	}
	go func() {
		code("Request", sess.Request(rank, goName, b)) // FIXME: pass error code to done
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuRequestVersion
func GoKungfuRequestVersion(rank int, version, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	goVersion := C.GoString(version)
	goName := C.GoString(name) // copy *C.char into go string before entering goroutine
	b := toVector(buf, count, dtype)
	if done == nil {
		return code("Pull", sess.Pull(rank, goVersion, goName, b))
	}
	go func() {
		code("Pull", sess.Pull(rank, goVersion, goName, b))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuSave
func GoKungfuSave(name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	goName := C.GoString(name) // copy *C.char into go string before entering goroutine
	b := toVector(buf, count, dtype)
	if done == nil {
		return code("Save", sess.Save(goName, b))
	}
	go func() {
		code("Save", sess.Save(goName, b))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuSaveVersion
func GoKungfuSaveVersion(version, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	goVersion := C.GoString(version)
	goName := C.GoString(name) // copy *C.char into go string before entering goroutine
	b := toVector(buf, count, dtype)
	if done == nil {
		return code("Save", kungfu.Save(goVersion, goName, b))
	}
	go func() {
		code("Save", kungfu.Save(goVersion, goName, b))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
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
	if done == nil {
		return code("AllReduce", sess.AllReduce(w))
	}
	go func() {
		code("AllReduce", sess.AllReduce(w))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
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
	if done == nil {
		return code("Reduce", sess.Reduce(w))
	}
	go func() {
		code("Reduce", sess.Reduce(w))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuBroadcast
func GoKungfuBroadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		// OP:      0, // FIXME: assert that OP is not used
		Name: C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		return code("Broadcast", sess.Broadcast(w))
	}
	go func() {
		code("Broadcast", sess.Broadcast(w))
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
		SendBuf: toVector(sendBuf, sendCount, sendDtype),
		RecvBuf: toVector(recvBuf, recvCount, recvDtype),
		// OP:      0, // FIXME: assert that OP is not used
		Name: C.GoString(name),
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		return code("Gather", sess.Gather(w))
	}
	go func() {
		code("Gather", sess.Gather(w))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoKungfuGetPeerLatencies
func GoKungfuGetPeerLatencies(recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype) int {
	results := toVector(recvBuf, recvCount, recvDtype).AsF32()
	sess := kungfu.CurrentSession()
	latencies := sess.GetPeerLatencies()
	// FIXME: check length
	for i := range results {
		results[i] = float32(latencies[i])
	}
	return 0
}

//export GoKungfuGetStrategyFromEnv
func GoKungfuGetStrategyFromEnv() C.KungFu_AllReduceStrategy {
	name := os.Getenv(kb.AllReduceStrategyEnvKey)
	return C.KungFu_AllReduceStrategy(kb.ParseStrategy(name))
}

func main() {}

func toVector(ptr unsafe.Pointer, count int, dtype C.KungFu_Datatype) *kb.Vector {
	if ptr == nil {
		if count > 0 {
			utils.ExitErr(fmt.Errorf("toVector: ptr is nil but count = %d", count))
		}
	}
	dt := kb.DataType(dtype)
	size := count * dt.Size()
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  size,
		Cap:  size,
	}
	return &kb.Vector{
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

func code(name string, err error) int {
	if err == nil {
		return 0
	}
	log.Errorf("kungfu operation %s failed: %v", name, err)
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1 // the caller should exit(1)
}
