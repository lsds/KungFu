package session

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func (sess *Session) AllReduce(w kb.Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *Session) AllReduceWith(tree []int32, w kb.Workspace) error {
	bg := plan.NewGraphFromTreeArray(tree)
	rg := plan.GenDefaultReduceGraph(bg)
	s0 := strategy{reduceGraph: rg, bcastGraph: bg}
	return sess.runStrategies(w, plan.EvenPartition, []strategy{s0})
}
