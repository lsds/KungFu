package kungfu

import (
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

func (sess *session) GetPeerLatencies() []time.Duration {
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
	conn := rch.NewPingConnection(plan.NetAddr(self), plan.NetAddr(peer))
	defer conn.Close()
	t0 := time.Now()
	var empty rch.Message
	conn.Send("ping", empty, rch.NoFlag)
	conn.Read("ping", empty)
	// FIXME: handle timeout
	return time.Since(t0)
}
