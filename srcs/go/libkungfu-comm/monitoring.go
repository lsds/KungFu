package main

/*
#include <kungfu/callback.h>
#include <kungfu/dtype.h>
#include <kungfu/op.h>
#include <kungfu/strategy.h>
*/
import "C"
import "unsafe"

//export GoKungfuGetPeerLatencies
func GoKungfuGetPeerLatencies(recvBuf unsafe.Pointer, recvCount int, recvDtype C.KungFu_Datatype) int {
	results := toVector(recvBuf, recvCount, recvDtype).AsF32()
	sess := defaultPeer.CurrentSession()
	latencies := sess.GetPeerLatencies()
	// FIXME: check length
	for i := range results {
		results[i] = float32(latencies[i])
	}
	return 0
}

//export GoKungfuCheckInterference
func GoKungfuCheckInterference() int {
	sess := defaultPeer.CurrentSession()
	if sess.CheckInterference() {
		return 1
	}
	return 0
}

//export GoCalcStats
func GoCalcStats() {
	sess := defaultPeer.CurrentSession()
	sess.CalcStats()
}

//export GoLogStats
func GoLogStats() {
	sess := defaultPeer.CurrentSession()
	sess.LogStats()
}

//export GoPrintStategyStats
func GoPrintStategyStats() {
	sess := defaultPeer.CurrentSession()
	sess.PrintStategyStats()
}
