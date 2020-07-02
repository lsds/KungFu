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
func GoKungfuResizeClusterFromURL(c *C.struct_peer_s, pChanged, pKeep *C.char) int {
	changed, keep, err := toPeer(c).ResizeClusterFromURL()
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pKeep = boolToChar(keep)
	return 0
}

//export GoKungfuProposeNewSize
func GoKungfuProposeNewSize(c *C.struct_peer_s, newSize int) int {
	err := toPeer(c).ProposeNewSize(newSize)
	if err != nil {
		log.Warnf("ProposeNewSize failed: %v", err)
	}
	return errorCode("ProposeNewSize", err)
}

//export GoKungfuSetTree
func GoKungfuSetTree(c *C.struct_peer_s, pTree unsafe.Pointer) int {
	sess := toPeer(c).CurrentSession()
	tree := toVector(pTree, sess.Size(), C.KungFu_INT32) // TODO: ensure pTree has size np in C++
	return callOP("SimpleSetGlobalStrategy", func() error { return sess.SimpleSetGlobalStrategy(tree.AsI32()) }, nil)
}
