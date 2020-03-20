package main

import (
	"fmt"
	"reflect"
	"unsafe"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
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
func GoKungfuInit() int {
	var err error
	kungfu, err = kf.New()
	if err != nil {
		return errorCode("New", err)
	}
	return errorCode("Start", kungfu.Start())
}

//export GoKungfuFinalize
func GoKungfuFinalize() int {
	return errorCode("Close", kungfu.Close())
}

//export GoKungfuUID
func GoKungfuUID() uint64 {
	return kungfu.UID()
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

//export GoKungfuLocalRank
func GoKungfuLocalRank() int {
	sess := kungfu.CurrentSession()
	return sess.LocalRank()
}

//export GoKungfuRequest
func GoKungfuRequest(rank int, pName *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	name := C.GoString(pName) // copy *C.char into go string before entering closure
	sess := kungfu.CurrentSession()
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := sess.Request(rank, "", name, b)
		if !ok {
			log.Warnf("Request %s not found", name)
		}
		return err
	}
	return callOP("Request", op, done)
}

//export GoKungfuRequestVersion
func GoKungfuRequestVersion(rank int, version, pName *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	name := C.GoString(pName) // copy *C.char into go string before entering closure
	sess := kungfu.CurrentSession()
	goVersion := C.GoString(version)
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := sess.Request(rank, goVersion, name, b)
		if !ok {
			log.Warnf("RequestVersion %s@%s not found", name, goVersion)
		}
		return err
	}
	return callOP("RequestVersion", op, done)
}

//export GoKungfuSave
func GoKungfuSave(name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	goName := C.GoString(name)
	b := toVector(buf, count, dtype)
	op := func() error { return kungfu.Save(goName, b) }
	return callOP("Save", op, done)
}

//export GoKungfuSaveVersion
func GoKungfuSaveVersion(version, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	goVersion := C.GoString(version)
	goName := C.GoString(name)
	b := toVector(buf, count, dtype)
	op := func() error { return kungfu.SaveVersion(goVersion, goName, b) }
	return callOP("SaveVersion", op, done)
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

func main() {
	fmt.Printf("%s is a library\n", utils.ProgName())
}

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

func callOP(name string, op func() error, done *C.callback_t) int {
	if done == nil {
		if kc.EnableStallDetection {
			defer utils.InstallStallDetector(name).Stop()
		}
		return errorCode(name, op())
	}
	go func() {
		if kc.EnableStallDetection {
			defer utils.InstallStallDetector(name).Stop()
		}
		errorCode(name, op()) // FIXME: pass error code to done
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

func errorCode(name string, err error) int {
	if err == nil {
		return 0
	}
	log.Errorf("kungfu operation %s failed: %v", name, err)
	return 1 // the caller should exit(1)
}
