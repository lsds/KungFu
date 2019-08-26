package main

import (
	"unsafe"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

// #include <kungfu.h>
import "C"

//export GoNewSharedVariable
func GoNewSharedVariable(name *C.char, count int, dtype C.KungFu_Datatype) int {
	return kungfu.CreateSharedVariable(C.GoString(name), count, kb.KungFu_Datatype(dtype))
}

//export GoGetSharedVariable
func GoGetSharedVariable(name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype) int {
	return kungfu.GetSharedVariable(C.GoString(name), toBuffer(buf, count, dtype))
}

//export GoPutSharedVariable
func GoPutSharedVariable(name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype) int {
	return kungfu.PutSharedVariable(C.GoString(name), toBuffer(buf, count, dtype))
}

//export GoAddSharedVariable
func GoAddSharedVariable(name *C.char, buf unsafe.Pointer, count int, dtype C.KungFu_Datatype) int {
	return kungfu.AddSharedVariable(C.GoString(name), toBuffer(buf, count, dtype))
}
