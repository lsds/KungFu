package session

import (
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/graph"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func (sess *Session) AllReduce(w base.Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.globalStrategies)
}

//AllReduceWith persoms an AllReduce collective communication operation
//given a tree topology for the strategy to be executted.
//ATTENTION: not stable feauture. Only for internal use.
func (sess *Session) AllReduceWith(tree []int32, w kb.Workspace) error {
	//TODO: decide whether the strategy created here should be stored
	//in the session object

	//ATTENTION: not stable, internal experimental use only
	var sl strategyList

	if len(tree) > 0 {
		bg, m, ok := graph.FromForestArrayI32(tree)
		assert.True(m == 1)
		assert.True(ok)
		sl = simpleSingleGraphStrategy(bg)
	} else {
		sl = sess.globalStrategies
	}

	return sess.runMonitoredStrategies(w, plan.EvenPartition, sl)
}

// CrossAllReduce performs allreduce across all local roots.
func (sess *Session) CrossAllReduce(w base.Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.crossStrategies)
}
