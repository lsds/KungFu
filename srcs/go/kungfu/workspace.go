package kungfu

import (
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
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

type shardHashFunc func(int, string) int

func simpleHash(i int, name string) int {
	return i
}

func nameBasedHash(i int, name string) int {
	var h int
	for _, c := range name {
		h += int(c) * int(c)
	}
	return h
}

func getshardHash() shardHashFunc {
	if kc.ShardHashMethod == `NAME` {
		log.Debugf("using name based hash")
		return nameBasedHash
	}
	return simpleHash
}
