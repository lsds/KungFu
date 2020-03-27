package session

import (
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Workspace contains the data that a Kungfu operation will be performed on.
type Workspace struct {
	SendBuf *kb.Vector
	RecvBuf *kb.Vector // TODO: if nil, will use SendBuf as in-place result
	OP      kb.OP
	Name    string
}

// 0 <= begin < end <= count - 1
func (w Workspace) slice(begin, end int) Workspace {
	var recvBuf *kb.Vector
	if w.RecvBuf != nil {
		recvBuf = w.RecvBuf.Slice(begin, end)
	}
	return Workspace{
		SendBuf: w.SendBuf.Slice(begin, end),
		RecvBuf: recvBuf,
		OP:      w.OP,
		Name:    fmt.Sprintf("part::%s[%d:%d]", w.Name, begin, end),
	}
}

// partitionFunc is the signature of function that parts the interval
type partitionFunc func(r plan.Interval, k int) []plan.Interval

func (w Workspace) split(p partitionFunc, k int) []Workspace {
	var ws []Workspace
	for _, r := range p(plan.Interval{Begin: 0, End: w.SendBuf.Count}, k) {
		ws = append(ws, w.slice(r.Begin, r.End))
	}
	return ws
}

// A strategyHashFunc is to map a given Workspace to a communication strategy.
// KungFu can create multiple communication strategies to balance the workload in the network core and edge links.
// We support hash the Workspaces based on their names or their partition IDs if they are chunked.
type strategyHashFunc func(int, string) uint64

func simpleHash(i int, name string) uint64 {
	return uint64(i)
}

func nameBasedHash(i int, name string) uint64 {
	var h uint64
	for _, c := range name {
		h += uint64(c) * uint64(c)
	}
	return h
}

func getStrategyHash() strategyHashFunc {
	if config.StrategyHashMethod == `NAME` {
		log.Debugf("using name based hash")
		return nameBasedHash
	}
	return simpleHash
}
