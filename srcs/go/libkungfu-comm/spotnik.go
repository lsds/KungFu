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
#include <stdint.h>
*/
import "C"

//export GoSpotnikAllReduce
func GoSpotnikAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, succeeded *C.int32_t, op C.KungFu_Op, pName *C.char, done *C.callback_t) int {
	name := C.GoString(pName)
	w := kf.Workspace{
		SendBuf: toVector(sendBuf, count, dtype),
		RecvBuf: toVector(recvBuf, count, dtype),
		OP:      kb.OP(op),
		Name:    name,
	}
	sess := kungfu.CurrentSession()
	if done == nil {
		code := errorCode("AllReduce", sess.AllReduce(w))
		*succeeded = C.int32_t(code)
		return code
	}
	go func() {
		*succeeded = C.int32_t(errorCode("AllReduce", sess.AllReduce(w)))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}

//export GoSpotnikRequest
func GoSpotnikRequest(rank int, name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype, succeeded *C.int32_t, done *C.callback_t) int {
	sess := kungfu.CurrentSession()
	goName := C.GoString(name) // copy *C.char into go string before entering closure
	b := toVector(buf, count, dtype)
	op := func() error {
		ok, err := sess.Request(rank, "", goName, b)
		if !ok {
			// log.Warnf("Request %s not found", goName) TODO
		}
		return err
	}
	if done == nil {
		code := errorCode(goName, op())
		*succeeded = C.int32_t(code)
		return code
	}
	go func() {
		*succeeded = C.int32_t(errorCode(goName, op()))
		C.invoke_callback(done)
		C.delete_callback(done)
	}()
	return 0
}
