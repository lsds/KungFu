package plan

import (
	"testing"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func Test_graph(t *testing.T) {
	n := 4
	hosts := fakeHosts(n)

	k := n * 4
	peers := hosts.genPeerList(k, DefaultPortRange)

	bcastGraph := GenTree(peers)
	reduceGraph := GenDefaultReduceGraph(bcastGraph)

	reduceGraph.Debug()
	bcastGraph.Debug()
	// TODO: add tests
}

func Test_NewGraphFromForestArray(t *testing.T) {
	{
		f := []int32{1, 1, 1}
		g, m, ok := NewGraphFromForestArray(f)
		assert.True(g != nil)
		assert.True(ok)
		assert.True(m == 1)
	}
	{
		f := []int32{0, 1, 2}
		g, m, ok := NewGraphFromForestArray(f)
		assert.True(g != nil)
		assert.True(ok)
		assert.True(m == 3)
	}
	{
		f := []int32{1, 2, 3}
		g, m, ok := NewGraphFromForestArray(f)
		assert.True(g == nil)
		assert.True(!ok)
		assert.True(m == 0)
	}
}
