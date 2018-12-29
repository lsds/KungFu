package main

import (
	"os"

	kf "github.com/luomai/kungfu/srcs/go/kungfu"
	"github.com/luomai/kungfu/srcs/go/log"
	"github.com/luomai/kungfu/srcs/go/wire"
)

// #include <kungfu.h>
// #include <callback.h>
import "C"

var kungfu *kf.Kungfu

//export GoKungfuInit
func GoKungfuInit(algo C.KungFu_AllReduceAlgo) int {
	log.Infof("GoKungfuInit")
	var err error
	config := kf.Config{Algo: wire.KungFu_AllReduceAlgo(algo)}
	kungfu, err = kf.New(config)
	if err != nil {
		exitErr(err)
	}
	return kungfu.Start()
}

//export GoKungfuFinalize
func GoKungfuFinalize() int {
	log.Infof("GoKungfuFinalize")
	return kungfu.Close()
}

//export GoKungfuNegotiate
func GoKungfuNegotiate(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string) int {
	return kungfu.Negotiate(sendBuf, recvBuf, count, wire.KungFu_Datatype(dtype), wire.KungFu_Op(op), name)
}

//export GoKungfuNegotiateAsync
func GoKungfuNegotiateAsync(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string, done *C.callback_t) int {
	name = string([]byte(name)) // TODO: verify that name is cloned
	go func() {
		GoKungfuNegotiate(sendBuf, recvBuf, count, dtype, op, name)
		if done != nil {
			C.invoke_callback(done)
			C.delete_callback(done)
		} else {
			log.Warnf("done is nil!")
		}
	}()
	return 0
}

func main() {}

func exitErr(err error) {
	log.Errorf("exit on error: %v", err)
	os.Exit(1)
}
