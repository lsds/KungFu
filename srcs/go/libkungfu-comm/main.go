package main

import (
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

//export GoKungfuNegotiate
func GoKungfuNegotiate(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char) int {
	return kungfu.Negotiate(
		toSlice(sendBuf, count, dtype),
		toSlice(recvBuf, count, dtype),
		count,
		kb.KungFu_Datatype(dtype),
		kb.KungFu_Op(op),
		C.GoString(name),
	)
}

//export GoKungfuNegotiateAsync
func GoKungfuNegotiateAsync(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	go func(name string) {
		kungfu.Negotiate(
			toSlice(sendBuf, count, dtype),
			toSlice(recvBuf, count, dtype),
			count,
			kb.KungFu_Datatype(dtype),
			kb.KungFu_Op(op),
			name,
		)
		if done != nil {
			C.invoke_callback(done)
			C.delete_callback(done)
		} else {
			log.Warnf("done is nil!")
		}
	}(C.GoString(name))
	return 0
}

//export GoKungfuGetAlgoFromEnv
func GoKungfuGetAlgoFromEnv() C.KungFu_AllReduceAlgo {
	name := os.Getenv(kb.AllReduceAlgoEnvKey)
	return C.KungFu_AllReduceAlgo(kb.ParseAlgo(name))
}

func main() {}

func toSlice(ptr unsafe.Pointer, count int, dtype C.KungFu_Datatype) []byte {
	size := count * kb.KungFu_Datatype(dtype).Size()
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  size,
		Cap:  size,
	}
	return *(*[]byte)(unsafe.Pointer(sh))
}
