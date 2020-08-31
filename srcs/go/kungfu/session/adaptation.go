package session

import (
	"github.com/lsds/KungFu/srcs/go/plan/graph"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func (sess *Session) SetGlobalStrategy(sl strategyList) error {
	sess.Lock()
	defer sess.Unlock()
	assert.OK(sess.barrier())

	ok, err := sess.BytesConsensus(sl.digestBytes(), "kungfu::SetStrategy")
	assert.True(ok)
	assert.OK(err)
	sess.globalStrategies = sl

	assert.OK(sess.barrier())
	return nil
}

func (sess *Session) SimpleSetGlobalStrategy(forest []int32) error {
	assert.True(len(forest) == len(sess.peers))
	bg, m, ok := graph.FromForestArrayI32(forest)
	assert.True(m == 1)
	assert.True(ok)
	return sess.SetGlobalStrategy(simpleSingleGraphStrategy(bg))
}
