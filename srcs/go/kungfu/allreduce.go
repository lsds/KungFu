package kungfu

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/luomai/kungfu/srcs/go/algo"
	"github.com/luomai/kungfu/srcs/go/log"
	rch "github.com/luomai/kungfu/srcs/go/rchannel"
	"github.com/luomai/kungfu/srcs/go/wire"
)

func divide(a, b int) (int, int) {
	q := a / b
	r := a - b*q
	return q, r
}

// rootedAllReduceFunc is the signature of allReduce algorithms that has a central root node
type rootedAllReduceFunc func(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, root int, name string) error

func (kf *Kungfu) simpleAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, root int, name string) error {
	if err := kf.reduce(sendBuf, recvBuf, count, dtype, op, root, name); err != nil {
		return err
	}
	return kf.bcast(recvBuf, count, dtype, root, name)
}

func (kf *Kungfu) circularAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, root int, name string) error {
	n := count * dtype.Size()

	sendTo := func(rank int) {
		task := kf.cluster.GetPeer(rank)
		m := rch.NewMessage(sendBuf[:n])
		kf.router.Send(task.NetAddr.WithName(name), *m)
	}

	recvAdd := func(rank int) {
		task := kf.cluster.GetPeer(rank)
		addr := task.NetAddr
		var m rch.Message
		kf.router.Recv(addr.WithName(name), &m)
		if int(m.Length) != n {
			// FIXME: don't panic
			panic("unexpected recv length")
		}
		algo.AddBy(recvBuf[:n], m.Data, count, dtype, op)
	}

	recvAssign := func(rank int) {
		task := kf.cluster.GetPeer(rank)
		addr := task.NetAddr
		var m rch.Message
		kf.router.Recv(addr.WithName(name), &m)
		if int(m.Length) != n {
			// FIXME: don't panic
			panic("unexpected recv length")
		}
		copy(recvBuf[:n], m.Data)
	}

	// k := len(cluster.Peers) // k >= 3
	myRank := kf.cluster.MyRank()
	myPrev, myNext := kf.cluster.PrevAndNext(myRank)
	rootPrev, rootNext := kf.cluster.PrevAndNext(root)

	R := func() { recvAdd(myPrev) }
	r := func() { recvAssign(myPrev) }
	S := func() { sendTo(myNext) }

	switch myRank {
	case root:
		{
			// RS
			R()
			S()
		}
	case rootPrev:
		{
			// RS|r
			R()
			S()
			r()
		}
	case rootNext:
		{
			// S|rS
			S()
			r()
			S()
		}
	default:
		{
			// RS|rS
			R()
			S()
			r()
			S()
		}
	}
	return nil
}

func (kf *Kungfu) treeAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, name string) error {
	n := count * dtype.Size()

	sendTo := func(rank int) {
		task := kf.cluster.GetPeer(rank)
		m := rch.NewMessage(sendBuf[:n])
		kf.router.Send(task.NetAddr.WithName(name), *m)
	}

	var lock sync.Mutex
	recvAdd := func(rank int) {
		task := kf.cluster.GetPeer(rank)
		addr := task.NetAddr
		var m rch.Message
		kf.router.Recv(addr.WithName(name), &m)
		lock.Lock()
		algo.AddBy(recvBuf[:n], m.Data, count, dtype, op)
		lock.Unlock()
	}

	recvAssign := func(rank int) {
		task := kf.cluster.GetPeer(rank)
		addr := task.NetAddr
		var m rch.Message
		kf.router.Recv(addr.WithName(name), &m)

		copy(recvBuf[:n], m.Data)
	}

	prun := func(ranks []int, op func(int)) {
		var wg sync.WaitGroup
		for _, rank := range ranks {
			wg.Add(1)
			go func(rank int) {
				op(rank)
				wg.Done()
			}(rank)
		}
		wg.Wait()
	}

	myRank := kf.cluster.MyRank()
	n1, n2, n3, n4 := kf.cluster.Neighbours(myRank)
	prun(n1, recvAdd)
	prun(n2, sendTo)
	prun(n3, recvAssign)
	prun(n4, sendTo)
	return nil
}

// symmetricAllReduce parts the original data into k parts and apply an rooted allReduce stratrgy to each part
func (kf *Kungfu) symmetricAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, name string, allReduce rootedAllReduceFunc) error {
	k := kf.cluster.Size()
	quo, rem := divide(count, k)

	var wg sync.WaitGroup
	var failed int32

	var offset int
	for i := 0; i < k; i++ {
		blockCount := func() int {
			if i < rem {
				return quo + 1
			}
			return quo
		}()
		wg.Add(1)
		go func(i int, offset, blockCount int) {
			x := offset * dtype.Size()
			y := (offset + blockCount) * dtype.Size()
			fullName := name + fmt.Sprintf(":part=%d", i) // TODO: use tag
			if err := allReduce(sendBuf[x:y], recvBuf[x:y], blockCount, dtype, op, i, fullName); err != nil {
				log.Warnf("part %d failed: %v", i, err)
				atomic.AddInt32(&failed, 1)
			}
			wg.Done()
		}(i, offset, blockCount)
		offset += blockCount
	}
	if offset != count {
		panic("invalid partition")
	}
	wg.Wait()
	if failed > 0 {
		return fmt.Errorf("%d parts failed", failed)
	}
	return nil
}

func (kf *Kungfu) fullSymmetricAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, name string) error {
	return kf.symmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name, kf.simpleAllReduce)
}

func (kf *Kungfu) ringAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, name string) error {
	k := kf.cluster.Size()
	if k < 3 {
		return fmt.Errorf("ringAllReduce requires k >= 3, but k=%d", k)
	}
	return kf.symmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name, kf.circularAllReduce)
}
