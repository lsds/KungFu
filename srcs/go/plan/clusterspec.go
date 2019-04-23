package plan

type ClusterSpec struct {
	Peers []PeerSpec
}

func (cs ClusterSpec) String() string {
	return toString(cs)
}
