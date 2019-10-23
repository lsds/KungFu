package kungfubase

// #include "kungfu/strategy.h"
import "C"

type Strategy C.KungFu_AllReduceStrategy

const (
	Star       Strategy = C.KungFu_StarAllReduce
	Ring       Strategy = C.KungFu_RingAllReduce
	Clique     Strategy = C.KungFu_CliqueAllReduce
	Tree       Strategy = C.KungFu_TreeAllReduce
	BinaryTree Strategy = C.KungFu_BinaryTreeAllReduce
)

var (
	strategyNames = map[Strategy]string{
		Star:       `STAR`,
		Ring:       `RING`,
		Clique:     `CLIQUE`,
		Tree:       `TREE`,
		BinaryTree: `BINARY_TREE`,
	}

	defaultStrategy = Tree
)

func StrategyNames() []string {
	var names []string
	for _, name := range strategyNames {
		names = append(names, name)
	}
	return names
}

func (s Strategy) String() string {
	return strategyNames[s]
}

func ParseStrategy(s string) Strategy {
	for k, v := range strategyNames {
		if s == v {
			return k
		}
	}
	return defaultStrategy
}
