package execution

import (
	"sync"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type PeerFunc func(plan.PeerID) error

// Par runs a function for a list of Peers in parallel
func (f PeerFunc) Par(ps plan.PeerList) error {
	errs := make([]error, len(ps))
	var wg sync.WaitGroup
	for i, p := range ps {
		wg.Add(1)
		go func(i int, p plan.PeerID) {
			errs[i] = f(p)
			wg.Done()
		}(i, p)
	}
	wg.Wait()
	return utils.MergeErrors(errs, "par")
}

// Seq runs a function for a list of Peers sequentially
func (f PeerFunc) Seq(ps plan.PeerList) error {
	for _, p := range ps {
		if err := f(p); err != nil {
			return err
		}
	}
	return nil
}
