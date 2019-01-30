package kungfu

import (
	"fmt"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Workspace contains the data that a Kungfu operation will be performed on.
type Workspace struct {
	SendBuf []byte
	RecvBuf []byte // TODO: if nil, will use SendBuf as in-place result
	Count   int
	Dtype   kb.KungFu_Datatype
	OP      kb.KungFu_Op
	Name    string
}

// 0 <= begin < end <= count - 1
func (w Workspace) slice(begin, end int) Workspace {
	i := begin * w.Dtype.Size()
	j := end * w.Dtype.Size()
	var recvBuf []byte
	if w.RecvBuf != nil {
		recvBuf = w.RecvBuf[i:j]
	}
	return Workspace{
		SendBuf: w.SendBuf[i:j],
		RecvBuf: recvBuf,
		Count:   end - begin,
		Dtype:   w.Dtype,
		OP:      w.OP,
		Name:    fmt.Sprintf("part::%s[%d:%d]", w.Name, begin, end),
	}
}

// partitionFunc is the signature of function that parts the interval
type partitionFunc func(r plan.Interval, k int) []plan.Interval

func (w Workspace) split(p partitionFunc, k int) []Workspace {
	var ws []Workspace
	for _, r := range p(plan.Interval{Begin: 0, End: w.Count}, k) {
		ws = append(ws, w.slice(r.Begin, r.End))
	}
	return ws
}
