package main

import "github.com/lsds/KungFu/srcs/go/utils"

import "C"

//export GoKungfuResizeCluster
func GoKungfuResizeCluster(pInitStep *C.char, size int, pChanged, pKeep *C.char) int {
	initStep := C.GoString(pInitStep)
	changed, keep, err := kungfu.ResizeCluster(initStep, size)
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pKeep = boolToChar(keep)
	return 0
}
