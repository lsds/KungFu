package plan

import "encoding/json"

type ClusterSpec struct {
	Peers []TaskSpec
}

func (cs ClusterSpec) String() string {
	bs, err := json.Marshal(cs)
	if err != nil {
		return ""
	}
	return string(bs)
}

func (cs ClusterSpec) ToProcSpec(rank int) ProcSpec {
	return ProcSpec{
		ClusterSpec: cs,
		SelfRank:    rank,
	}
}
