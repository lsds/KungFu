package main

import (
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
)

/*
#include <kungfu/dtype.h>
*/
import "C"

//export GoKungfuResizeClusterFromURL
func GoKungfuResizeClusterFromURL(pChanged, pKeep *C.char) int {
	changed, keep, err := defaultPeer.ResizeClusterFromURL()
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pKeep = boolToChar(keep)
	return 0
}

//export GoKungfuProposeNewSize
func GoKungfuProposeNewSize(newSize int) int {
	err := defaultPeer.ProposeNewSize(newSize)
	if err != nil {
		log.Warnf("ProposeNewSize failed: %v", err)
	}
	return errorCode("ProposeNewSize", err)
}

//export GoKungfuSetTree
func GoKungfuSetTree(pTree unsafe.Pointer) int {
	sess := defaultPeer.CurrentSession()
	tree := toVector(pTree, sess.Size(), C.KungFu_INT32) // TODO: ensure pTree has size np in C++
	return callOP("Barrier", func() error { return sess.SimpleSetStrategy(tree.AsI32()) }, nil)
}
