package kungfubase

// #include "strategy.h"
import "C"

type Strategy C.strategy

const (
	KungFu_Star   Strategy = C.star
	KungFu_Ring   Strategy = C.ring
	KungFu_Clique Strategy = C.clique
	KungFu_Tree   Strategy = C.tree
)

var (
	strategyNames = map[Strategy]string{
		KungFu_Star:   `STAR`,
		KungFu_Ring:   `RING`,
		KungFu_Clique: `CLIQUE`,
		KungFu_Tree:   `TREE`,
	}

	defaultStrategy = KungFu_Tree
)

func StrategyNames() []string {
	var names []string
	for _, name := range strategyNames {
		names = append(names, name)
	}
	return names
}

func (a Strategy) String() string {
	for k, v := range strategyNames {
		if a == k {
			return v
		}
	}
	return strategyNames[defaultStrategy]
}

func ParseStrategy(s string) Strategy {
	for k, v := range strategyNames {
		if s == v {
			return k
		}
	}
	return defaultStrategy
}
