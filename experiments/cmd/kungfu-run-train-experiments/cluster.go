package main

import (
	"sort"

	"github.com/lsds/KungFu/srcs/go/plan"
)

type Cluster struct {
	Hostlist plan.HostList
	Size     int
}

func generateClusters(hl plan.HostList, sizes []int) []Cluster {
	sort.Sort(sort.Reverse(sort.IntSlice(sizes)))
	var cs []Cluster
	for _, s := range sizes {
		c := Cluster{
			Hostlist: hl.ShrinkToFit(s),
			Size:     s,
		}
		cs = append(cs, c)
	}
	return cs
}
