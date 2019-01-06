package main

import (
	kf "github.com/luomai/kungfu/srcs/go/kungfu"
	"github.com/luomai/kungfu/srcs/go/log"
	"github.com/luomai/kungfu/srcs/go/utils"
	"github.com/luomai/kungfu/srcs/go/wire"
)

// #include <kungfu.h>
// #include <callback.h>
import "C"

var kungfu *kf.Kungfu

//export GoKungfuInit
func GoKungfuInit(algo C.KungFu_AllReduceAlgo) int {
	var err error
	config := kf.Config{Algo: wire.KungFu_AllReduceAlgo(algo)}
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
func GoKungfuNegotiate(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char) int {
	return kungfu.Negotiate(sendBuf, recvBuf, count, wire.KungFu_Datatype(dtype), wire.KungFu_Op(op), C.GoString(name))
}

//export GoKungfuNegotiateAsync
func GoKungfuNegotiateAsync(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	go func(name string) {
		kungfu.Negotiate(sendBuf, recvBuf, count, wire.KungFu_Datatype(dtype), wire.KungFu_Op(op), name)
		if done != nil {
			C.invoke_callback(done)
			C.delete_callback(done)
		} else {
			log.Warnf("done is nil!")
		}
	}(C.GoString(name))
	return 0
}

func main() {}
