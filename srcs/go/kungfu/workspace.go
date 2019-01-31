package kungfu

import (
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Workspace contains the data that a Kungfu operation will be performed on.
type Workspace struct {
	SendBuf *kb.Buffer
	RecvBuf *kb.Buffer // TODO: if nil, will use SendBuf as in-place result
	OP      kb.KungFu_Op
	Name    string
}

// 0 <= begin < end <= count - 1
func (w Workspace) slice(begin, end int) Workspace {
	var recvBuf *kb.Buffer
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
