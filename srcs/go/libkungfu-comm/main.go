package main

import (
	"fmt"
	"reflect"
	"unsafe"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
)

/*
#include <kungfu/callback.h>
#include <kungfu/dtype.h>
#include <kungfu/op.h>
#include <kungfu/strategy.h>

struct peer_s {
    void *p;
};
*/
import "C"

func toPeer(c *C.struct_peer_s) *peer.Peer {
	return (*peer.Peer)(c.p)
}

//export GoKungfuInit
func GoKungfuInit(c *C.struct_peer_s) int {
	peer, err := peer.New()
	if err != nil {
		return errorCode("New", err)
	}
	c.p = unsafe.Pointer(peer)
	return errorCode("Start", peer.Start())
}

//export GoKungfuFinalize
func GoKungfuFinalize(c *C.struct_peer_s) int {
	return errorCode("Close", toPeer(c).Close())
}

//export GoKungfuDetached
func GoKungfuDetached(c *C.struct_peer_s) bool {
	return toPeer(c).Detached()
}

//export GoKungfuUID
func GoKungfuUID(c *C.struct_peer_s) uint64 {
	return toPeer(c).UID()
}

//export GoKungfuSize
func GoKungfuSize(c *C.struct_peer_s) int {
	sess := toPeer(c).CurrentSession()
	return sess.Size()
}

//export GoKungfuRank
func GoKungfuRank(c *C.struct_peer_s) int {
	sess := toPeer(c).CurrentSession()
	return sess.Rank()
}

//export GoKungfuLocalRank
func GoKungfuLocalRank(c *C.struct_peer_s) int {
	sess := toPeer(c).CurrentSession()
	return sess.LocalRank()
}

//export GoKungfuLocalSize
func GoKungfuLocalSize(c *C.struct_peer_s) int {
	sess := toPeer(c).CurrentSession()
	return sess.LocalSize()
}

//export GoKungfuHostCount
func GoKungfuHostCount(c *C.struct_peer_s) int {
	sess := toPeer(c).CurrentSession()
	return sess.HostCount()
}

//export GoKungfuRequest
func GoKungfuRequest(c *C.struct_peer_s, rank int, pName *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	name := C.GoString(pName) // copy *C.char into go string before entering closure
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := toPeer(c).RequestRank(rank, "", name, b)
		if !ok {
			log.Warnf("Request %s not found", name)
		}
		return err
	}
	return callOP("Request", op, done)
}

//export GoKungfuRequestVersion
func GoKungfuRequestVersion(c *C.struct_peer_s, rank int, version, pName *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	name := C.GoString(pName) // copy *C.char into go string before entering closure
	goVersion := C.GoString(version)
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := toPeer(c).RequestRank(rank, goVersion, name, b)
		if !ok {
			log.Warnf("RequestVersion %s@%s not found", name, goVersion)
		}
		return err
	}
	return callOP("RequestVersion", op, done)
}

//export GoKungfuSave
func GoKungfuSave(c *C.struct_peer_s, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	goName := C.GoString(name)
	b := toVector(buf, count, dtype)
	op := func() error { return toPeer(c).Save(goName, b) }
	return callOP("Save", op, done)
}

//export GoKungfuSaveVersion
func GoKungfuSaveVersion(c *C.struct_peer_s, version, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, done *C.callback_t) int {
	goVersion := C.GoString(version)
	goName := C.GoString(name)
	b := toVector(buf, count, dtype)
	op := func() error { return toPeer(c).SaveVersion(goVersion, goName, b) }
	return callOP("SaveVersion", op, done)
}

//export GoKungfuGetPeerLatencies
func GoKungfuGetPeerLatencies(c *C.struct_peer_s, recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype) int {
	results := toVector(recvBuf, recvCount, recvDtype).AsF32()
	sess := toPeer(c).CurrentSession()
	latencies := sess.GetPeerLatencies()
	// FIXME: check length
	for i := range results {
		results[i] = float32(latencies[i])
	}
	return 0
}

//export GoKungfuNoop
func GoKungfuNoop(done *C.callback_t) int {
	noop := func() error { return nil }
	return callOP("noop", noop, done)
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
