package kungfubase

// #include "kungfu/strategy.h"
import "C"

type Strategy C.KungFu_Strategy

const (
	Star           Strategy = C.KungFu_Star
	Ring           Strategy = C.KungFu_Ring
	Clique         Strategy = C.KungFu_Clique
	Tree           Strategy = C.KungFu_Tree
	BinaryTree     Strategy = C.KungFu_BinaryTree
	BinaryTreeStar Strategy = C.KungFu_BinaryTreeStar
	Auto           Strategy = C.KungFu_AUTO
)

var (
	strategyNames = map[Strategy]string{
		Star:           `STAR`,
		Ring:           `RING`,
		Clique:         `CLIQUE`,
		Tree:           `TREE`,
		BinaryTree:     `BINARY_TREE`,
		BinaryTreeStar: `BINARY_TREE_STAR`,
		Auto:           `AUTO`,
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
