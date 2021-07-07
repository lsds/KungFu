package main

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/env"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
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

var defaultPeer *peer.Peer

//export GoKungfuInit
func GoKungfuInit() int {
	var err error
	defaultPeer, err = peer.New()
	if err != nil {
		return errorCode("New", err)
	}
	return errorCode("Start", defaultPeer.Start())
}

//export GoKungfuInitFromJSON
func GoKungfuInitFromJSON(pJSON *C.char) int {
	js := C.GoString(pJSON)
	log.Errorf("GoKungfuInitFromJSON: %s", js)
	cfg, err := env.ParseConfigFromJSON(js)
	if err != nil {
		errorCode("ParseConfigFromJSON", err)
	}
	defaultPeer, err = peer.NewFromConfig(cfg)
	if err != nil {
		return errorCode("NewFromConfig", err)
	}
	return errorCode("Start", defaultPeer.Start())
}

//export GoKungfuInitSingleMachine
func GoKungfuInitSingleMachine(rank, size C.int) int {
	cfg, err := env.SingleMachineEnv(int(rank), int(size))
	if err != nil {
		return errorCode("SingleMachineEnv", err)
	}
	defaultPeer, err = peer.NewFromConfig(cfg)
	if err != nil {
		return errorCode("NewFromConfig", err)
	}
	return errorCode("Start", defaultPeer.Start())
}

//export GoKungfuFinalize
func GoKungfuFinalize() int {
	return errorCode("Close", defaultPeer.Close())
}

//export GoKungfuDetached
func GoKungfuDetached() bool {
	return defaultPeer.Detached()
}

//export GoKungfuUID
func GoKungfuUID() uint64 {
	return defaultPeer.UID()
}

//export GoKungfuSize
func GoKungfuSize() int {
	sess := defaultPeer.CurrentSession()
	return sess.Size()
}

//export GoKungfuRank
func GoKungfuRank() int {
	sess := defaultPeer.CurrentSession()
	return sess.Rank()
}

//export GoKungfuLocalRank
func GoKungfuLocalRank() int {
	sess := defaultPeer.CurrentSession()
	return sess.LocalRank()
}

//export GoKungfuLocalSize
func GoKungfuLocalSize() int {
	sess := defaultPeer.CurrentSession()
	return sess.LocalSize()
}

//export GoKungfuHostCount
func GoKungfuHostCount() int {
	sess := defaultPeer.CurrentSession()
	return sess.HostCount()
}

//export GoKungfuRequest
func GoKungfuRequest(rank int, pName *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	name := C.GoString(pName) // copy *C.char into go string before entering closure
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := defaultPeer.RequestRank(rank, "", name, b)
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
	goVersion := C.GoString(version)
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := defaultPeer.RequestRank(rank, goVersion, name, b)
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
	op := func() error { return defaultPeer.Save(goName, b) }
	return callOP("Save", op, done)
}

//export GoKungfuSaveVersion
func GoKungfuSaveVersion(version, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	goVersion := C.GoString(version)
	goName := C.GoString(name)
	b := toVector(buf, count, dtype)
	op := func() error { return defaultPeer.SaveVersion(goVersion, goName, b) }
	return callOP("SaveVersion", op, done)
}

//export GoKungfuNoop
func GoKungfuNoop(done *C.callback_t) int {
	noop := func() error { return nil }
	return callOP("noop", noop, done)
}

func main() {
	fmt.Printf("%s is a library\n", utils.ProgName())
}

func toVector(ptr unsafe.Pointer, count int, dtype C.KungFu_Datatype) *base.Vector {
	if ptr == nil {
		if count > 0 {
			utils.ExitErr(fmt.Errorf("toVector: ptr is nil but count = %d", count))
		}
	}
	dt := base.DataType(dtype)
	size := count * dt.Size()
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  size,
		Cap:  size,
	}
	return &base.Vector{
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
		if config.EnableStallDetection {
			defer utils.InstallStallDetector(name).Stop()
		}
		return errorCode(name, op())
	}
	go func() {
		if config.EnableStallDetection {
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
