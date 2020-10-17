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

//export GoKungfuResizeCluster
func GoKungfuResizeCluster(newSize int, pChanged, pDetached *C.char) int {
	changed, detached, err := defaultPeer.ResizeCluster(newSize)
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pDetached = boolToChar(detached)
	return 0
}

//export GoKungfuResizeClusterFromURL
func GoKungfuResizeClusterFromURL(pChanged, pDetached *C.char) int {
	changed, detached, err := defaultPeer.ResizeClusterFromURL()
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pDetached = boolToChar(detached)
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
	return callOP("SimpleSetGlobalStrategy", func() error { return sess.SimpleSetGlobalStrategy(tree.AsI32()) }, nil)
}
