package session

import (
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func (sess *Session) AllReduce(w base.Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.strategies)
}

func (sess *Session) AllReduceWith(forest []int32, w base.Workspace) error {
	bg, m, ok := plan.NewGraphFromForestArray(forest)
	assert.True(m == 1)
	assert.True(ok)
	rg := plan.GenDefaultReduceGraph(bg)
	s0 := strategy{reduceGraph: rg, bcastGraph: bg}
	return sess.runStrategies(w, plan.EvenPartition, []strategy{s0})
}
