package main

import "github.com/lsds/KungFu/srcs/go/plan"

type peerList map[string]plan.PeerSpec

func (p peerList) Sub(q peerList) []plan.PeerSpec {
	var d []plan.PeerSpec
	for k, v := range p {
		if _, ok := q[k]; !ok {
			d = append(d, v)
		}
	}
	return d
}
