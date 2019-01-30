package main

import (
	"fmt"
	"os"
	"reflect"
	"unsafe"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/profile"
	"github.com/lsds/KungFu/srcs/go/utils"
)

// #include <kungfu.h>
// #include <kungfu_base.h>
import "C"

var kungfu *kf.Kungfu

func deinit() {
	profile.Default.WriteSummary(os.Stdout)
	cluster := kungfu.DebugCurrentCluster()
	filename := fmt.Sprintf("peer-%02d-of-%02d.events.log", cluster.MyRank(), cluster.Size())
	f, err := os.Create(filename)
	if err != nil {
		return
	}
	defer f.Close()
	profile.Default.WriteEvents(f)
}

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
	defer deinit()
	return kungfu.Close()
}

//export GoKungfuAllReduce
func GoKungfuAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name *C.char, done *C.callback_t) int {
	w := kf.Workspace{
		SendBuf: toSlice(sendBuf, count, dtype),
		RecvBuf: toSlice(recvBuf, count, dtype),
		Count:   count,
		Dtype:   kb.KungFu_Datatype(dtype),
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
