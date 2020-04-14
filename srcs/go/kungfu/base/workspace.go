package base

import (
	"fmt"

	"github.com/lsds/KungFu/srcs/go/plan"
)

// Workspace contains the data that a Kungfu operation will be performed on.
type Workspace struct {
	SendBuf *Vector
	RecvBuf *Vector // TODO: if nil, will use SendBuf as in-place result
	OP      OP
	Name    string
}

// 0 <= begin < end <= count - 1
func (w Workspace) slice(begin, end int) Workspace {
	var recvBuf *Vector
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
type PartitionFunc func(r plan.Interval, k int) []plan.Interval

func (w Workspace) Split(p PartitionFunc, k int) []Workspace {
	var ws []Workspace
	for _, r := range p(plan.Interval{Begin: 0, End: w.SendBuf.Count}, k) {
		ws = append(ws, w.slice(r.Begin, r.End))
	}
	return ws
}
