package base

import "errors"

// #include "kungfu/strategy.h"
import "C"

type Strategy C.KungFu_Strategy

const (
	Star                Strategy = C.KungFu_Star
	MultiStar           Strategy = C.KungFu_MultiStar
	Ring                Strategy = C.KungFu_Ring
	Clique              Strategy = C.KungFu_Clique
	Tree                Strategy = C.KungFu_Tree
	BinaryTree          Strategy = C.KungFu_BinaryTree
	BinaryTreeStar      Strategy = C.KungFu_BinaryTreeStar
	MultiBinaryTreeStar Strategy = C.KungFu_MultiBinaryTreeStar
	Auto                Strategy = C.KungFu_AUTO
)

const DefaultStrategy = BinaryTreeStar

var (
	strategyNames = map[Strategy]string{
		Star:                `STAR`,
		MultiStar:           `MULTI_STAR`,
		Ring:                `RING`,
		Clique:              `CLIQUE`,
		Tree:                `TREE`,
		BinaryTree:          `BINARY_TREE`,
		BinaryTreeStar:      `BINARY_TREE_STAR`,
		MultiBinaryTreeStar: `MULTI_BINARY_TREE_STAR`,
		Auto:                `AUTO`,
	}
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

// Set implements flags.Value::Set
func (s *Strategy) Set(val string) error {
	value, err := ParseStrategy(val)
	if err != nil {
		return err
	}
	*s = *value
	return nil
}

var errInvalidStrategy = errors.New("invalid strategy")

func ParseStrategy(s string) (*Strategy, error) {
	for k, v := range strategyNames {
		if s == v {
			return &k, nil
		}
	}
	return nil, errInvalidStrategy
}
