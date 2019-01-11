package plan

import "testing"

func Test_graph(t *testing.T) {
	n := 4
	hosts := fakeHosts(n)

	k := n * 4
	tasks := genTaskSpecs(k, hosts)
	g1 := genDefaultGatherGraph(tasks)
	g2 := g1.Reverse()
	g1.Debug()
	g2.Debug()
	// TODO: add tests
}
