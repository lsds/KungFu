package main

import "github.com/lsds/KungFu/srcs/go/plan"

type peerList map[string]plan.PeerID

func (p peerList) Sub(q peerList) []plan.PeerID {
	var d []plan.PeerID
	for k, v := range p {
		if _, ok := q[k]; !ok {
			d = append(d, v)
		}
	}
	return d
}
