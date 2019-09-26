package fakemodel

import (
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

var Models = map[string][]int{
	"vgg16-imagenet":    vgg16Imagenet,
	"resnet50-imagenet": resnet50Imagenet,
	"slp-mnist":         slpMNIST,
}

var Names = func(m map[string][]int) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}(Models)

type doubleBuffer struct {
	SendBuf *kb.Buffer
	RecvBuf *kb.Buffer
}

func newDoubleBuffer(dtype kb.KungFu_Datatype, count int) doubleBuffer {
	return doubleBuffer{
		SendBuf: kb.NewBuffer(count, dtype),
		RecvBuf: kb.NewBuffer(count, dtype),
	}
}

type FakeModel struct {
	Names   []string
	Buffers map[string]doubleBuffer
}

func New(sizes []int, dtype kb.KungFu_Datatype) *FakeModel {
	var names []string
	buffers := make(map[string]doubleBuffer)
	for i, size := range sizes {
		name := fmt.Sprintf("NegotiatedGrad_%d/AllReduce", i)
		buffers[name] = newDoubleBuffer(dtype, size)
		names = append(names, name)
	}
	return &FakeModel{
		Names:   names,
		Buffers: buffers,
	}
}
