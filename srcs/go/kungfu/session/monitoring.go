package session

import (
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
)

func (sess *Session) GetPeerLatencies() []time.Duration {
	results := make([]time.Duration, len(sess.peers))
	var wg sync.WaitGroup
	for rank, peer := range sess.peers {
		if rank != sess.rank {
			wg.Add(1)
			go func(rank int, peer plan.PeerID) {
				results[rank] = getLatency(sess.self, peer)
				wg.Done()
			}(rank, peer)
		} else {
			results[rank] = 0
		}
	}
	wg.Wait()
	return results
}

func getLatency(self, peer plan.PeerID) time.Duration {
	client := client.New(self)
	d, err := client.Ping(peer)
	if err != nil {
		log.Errorf("ping(%s) failed, error ignored!", peer)
		// TODO handle error
	}
	return d
}
