package session

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func (sess *Session) AllReduce(w kb.Workspace) error {
	return sess.runStrategies(w, plan.EvenPartition, sess.strategies)
}

//AllReduceWith persoms an AllReduce collective communication operation
//given a tree topology for the strategy to be executted.
//ATTENTION: not stable feauture. Only for internal use.
func (sess *Session) AllReduceWith(tree []int32, w kb.Workspace) error {
	//TODO: decide whether the strategy created here should be stored
	//in the session object

	//ATTENTION: not stable, internal experimental use only
	var ss []strategy

	if len(tree) > 0 {
		bg := plan.NewGraphFromTreeArray(tree)
		rg := plan.GenDefaultReduceGraph(bg)
		ss = []strategy{{reduceGraph: rg, bcastGraph: bg}}
	} else {
		ss = sess.strategies
	}

	return sess.runMonitoresStrategies(w, plan.EvenPartition, ss)
}
