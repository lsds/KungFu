package kungfubase

// #include <kungfu.h>
import "C"

const (
	ClusterSpecEnvKey   = `KUNGFU_CLUSTER_SPEC`
	SelfRankEnvKey      = `KUNGFU_SELF_RANK`
	AllReduceAlgoEnvKey = `KUNGFU_ALLREDUCE_ALGO`
)

type KungFu_Datatype C.KungFu_Datatype

var (
	KungFu_INT32  = KungFu_Datatype(C.KungFu_INT32)
	KungFu_FLOAT  = KungFu_Datatype(C.KungFu_FLOAT)
	KungFu_DOUBLE = KungFu_Datatype(C.KungFu_DOUBLE)
)

func (dtype KungFu_Datatype) Size() int {
	return int(C.kungfu_type_size(C.KungFu_Datatype(dtype)))
}

type KungFu_Op C.KungFu_Op

var (
	KungFu_SUM = KungFu_Op(C.KungFu_SUM)
	KungFu_MIN = KungFu_Op(C.KungFu_MIN)
	KungFu_MAX = KungFu_Op(C.KungFu_MAX)
)

type KungFu_AllReduceAlgo C.KungFu_AllReduceAlgo

var (
	KungFu_Star   = KungFu_AllReduceAlgo(C.KungFu_StarAllReduce)
	KungFu_Ring   = KungFu_AllReduceAlgo(C.KungFu_RingAllReduce)
	KungFu_Clique = KungFu_AllReduceAlgo(C.KungFu_CliqueAllReduce)
	KungFu_Tree   = KungFu_AllReduceAlgo(C.KungFu_TreeAllReduce)

	algoNames = map[KungFu_AllReduceAlgo]string{
		KungFu_Star:   `STAR`,
		KungFu_Ring:   `RING`,
		KungFu_Clique: `CLIQUE`,
		KungFu_Tree:   `TREE`,
	}

	defaultAlgo = KungFu_Tree
)

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
