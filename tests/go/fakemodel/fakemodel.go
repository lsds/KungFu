package fakemodel

import (
	"fmt"
	"sort"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/tests/go/testutils"
)

var Models = map[string][]int{
	"resnet50-imagenet": resnet50Imagenet,
	"vgg16-imagenet":    vgg16Imagenet,
	"slp-mnist":         slpMNIST,
	"bert":              bert,
}

var Names = func(m map[string][]int) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}(Models)

func sum(a []int) int {
	var t int
	for _, x := range a {
		t += x
	}
	return t
}

type DoubleBuffer struct {
	SendBuf *kb.Vector
	RecvBuf *kb.Vector
}

func NewDoubleBuffer(dtype kb.DataType, count int) DoubleBuffer {
	return DoubleBuffer{
		SendBuf: kb.NewVector(count, dtype),
		RecvBuf: kb.NewVector(count, dtype),
	}
}

type FakeModel struct {
	Names   []string
	Buffers map[string]DoubleBuffer
}

func New(sizes []int, dtype kb.DataType, fuse bool) *FakeModel {
	if fuse {
		sizes = []int{sum(sizes)}
	}
	var names []string
	buffers := make(map[string]DoubleBuffer)
	for i, size := range sizes {
		name := fmt.Sprintf("NegotiatedGrad_%d/AllReduce", i)
		buffers[name] = NewDoubleBuffer(dtype, size)
		names = append(names, name)
	}
	return &FakeModel{
		Names:   names,
		Buffers: buffers,
	}
}

func (m *FakeModel) Size() int {
	var n int
	for _, b := range m.Buffers {
		n += len(b.SendBuf.Data)
	}
	return n
}

func (m *FakeModel) MinSize() int {
	s := len(m.Buffers[m.Names[0]].SendBuf.Data)
	for _, b := range m.Buffers {
		if n := len(b.SendBuf.Data); n < s {
			s = n
		}
	}
	return s
}

func (m *FakeModel) MaxSize() int {
	s := len(m.Buffers[m.Names[0]].SendBuf.Data)
	for _, b := range m.Buffers {
		if n := len(b.SendBuf.Data); n > s {
			s = n
		}
	}
	return s
}

func (m *FakeModel) Info() string {
	return fmt.Sprintf("%d parameters, size ~ [%s, %s], total %s", len(m.Buffers),
		testutils.ShowSize(int64(m.MinSize())),
		testutils.ShowSize(int64(m.MaxSize())),
		testutils.ShowSize(int64(m.Size())))
}

func (m *FakeModel) ShowInfo() {
	log.Infof("model has %s", m.Info())
}
