package main

import "github.com/lsds/KungFu/srcs/go/utils"

import "C"

//export GoKungfuResizeClusterFromURL
func GoKungfuResizeClusterFromURL(pChanged, pKeep *C.char) int {
	changed, keep, err := kungfu.ResizeClusterFromURL()
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pKeep = boolToChar(keep)
	return 0
}
