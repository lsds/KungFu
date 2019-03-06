package main

import (
	"unsafe"

	og "github.com/lsds/KungFu/srcs/go/ordergroup"
)

/*
#include <kungfu_base.h>

struct order_group_s {
    void * ptr;
};

*/
import "C"

//export GoNewRankedOrderGroup
func GoNewRankedOrderGroup(nNames int, cPtr *C.struct_order_group_s) {
	gPtr := og.NewRanked(nNames)
    // TODO: make sure gPtr will not be auto GCed.
	cPtr.ptr = unsafe.Pointer(gPtr)
}

//export GoFreeOrderGroup
func GoFreeOrderGroup(cPtr *C.struct_order_group_s) {
    freeOrderGroup(toOrderGroup(cPtr))
}

//export GoOrderGroupDoRank
func GoOrderGroupDoRank(cPtr *C.struct_order_group_s, rank int, task *C.callback_t) {
	g := toOrderGroup(cPtr)
	g.DoRank(rank, func() {
		C.invoke_callback(task)
		C.delete_callback(task)
	})
}

//export GoOrderGroupWait
func GoOrderGroupWait(cPtr *C.struct_order_group_s) {
	g := toOrderGroup(cPtr)
	g.Wait()
}

func freeOrderGroup(gPtr *og.OrderGroup) {
    // TODO:
}

func toOrderGroup(cPtr *C.struct_order_group_s) *og.OrderGroup {
	return (*og.OrderGroup)(cPtr.ptr)
}
