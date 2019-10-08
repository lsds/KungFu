package main

import (
	"unsafe"

	og "github.com/lsds/KungFu/srcs/go/ordergroup"
)

/*
#include <kungfu/callback.h>

struct order_group_s {
    void * _go_ptr;
};

*/
import "C"

//export GoNewOrderGroup
func GoNewOrderGroup(nNames int, cPtr *C.struct_order_group_s) {
	goPtr := og.New(nNames, og.Option{AutoWait: true})
	// TODO: make sure goPtr will not be auto GCed.
	cPtr._go_ptr = unsafe.Pointer(goPtr)
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

func freeOrderGroup(g *og.OrderGroup) {
	g.Stop()
	// TODO: GC g
}

func toOrderGroup(cPtr *C.struct_order_group_s) *og.OrderGroup {
	return (*og.OrderGroup)(cPtr._go_ptr)
}
