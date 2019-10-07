package main

import "github.com/lsds/KungFu/srcs/go/utils"

import "C"

//export GoKungfuResizeCluster
func GoKungfuResizeCluster(pCkpt *C.char, size int, pKeep *C.char) int {
	ckpt := C.GoString(pCkpt)
	keep, err := kungfu.ResizeCluster(ckpt, size)
	if err != nil {
		utils.ExitErr(err)
	}
	*pKeep = boolToChar(keep)
	return 0
}
