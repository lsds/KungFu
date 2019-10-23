package plan

import "testing"

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
