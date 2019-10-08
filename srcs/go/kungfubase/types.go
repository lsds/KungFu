package kungfubase

// #include "strategy.h"
import "C"

const (
	CheckpointEnvKey = `KUNGFU_INIT_CKPT`
	ParentIDEnvKey   = `KUNGFU_PARENT_ID`
	HostListEnvKey   = `KUNGFU_HOST_LIST`

	PeerListEnvKey      = `KUNGFU_INIT_PEERS`
	HostSpecEnvKey      = `KUNGFU_HOST_SPEC`
	SelfSpecEnvKey      = `KUNGFU_SELF_SPEC` // self spec should never change during the life of a process
	InitSessEnvKey      = `KUNGFU_INIT_SESS`
	InitStepEnvKey      = `KUNGFU_INIT_STEP`
	AllReduceAlgoEnvKey = `KUNGFU_ALLREDUCE_ALGO` // FIXME: remove it
)

type KungFu_AllReduceAlgo int

var (
	KungFu_Star   = KungFu_AllReduceAlgo(C.star)
	KungFu_Ring   = KungFu_AllReduceAlgo(C.ring)
	KungFu_Clique = KungFu_AllReduceAlgo(C.clique)
	KungFu_Tree   = KungFu_AllReduceAlgo(C.tree)

	algoNames = map[KungFu_AllReduceAlgo]string{
		KungFu_Star:   `STAR`,
		KungFu_Ring:   `RING`,
		KungFu_Clique: `CLIQUE`,
		KungFu_Tree:   `TREE`,
	}

	defaultAlgo = KungFu_Tree
)

func AllAlgoNames() []string {
	var names []string
	for _, name := range algoNames {
		names = append(names, name)
	}
	return names
}

func (a KungFu_AllReduceAlgo) String() string {
	for k, v := range algoNames {
		if a == k {
			return v
		}
	}
	return algoNames[defaultAlgo]
}

func ParseAlgo(s string) KungFu_AllReduceAlgo {
	for k, v := range algoNames {
		if s == v {
			return k
		}
	}
	return defaultAlgo
}
