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

type KungFu_Datatype DataType

var (
	KungFu_UINT8  = KungFu_Datatype(U8)
	KungFu_INT32  = KungFu_Datatype(I32)
	KungFu_INT64  = KungFu_Datatype(I64)
	KungFu_FLOAT  = KungFu_Datatype(F32)
	KungFu_DOUBLE = KungFu_Datatype(F64)
)

func (dtype KungFu_Datatype) Size() int {
	return DataType(dtype).Size()
}

type KungFu_Op OP

var (
	KungFu_SUM = KungFu_Op(SUM)
	KungFu_MIN = KungFu_Op(MIN)
	KungFu_MAX = KungFu_Op(MAX)
)

type KungFu_AllReduceAlgo int

// C.KungFu_AllReduceAlgo

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
