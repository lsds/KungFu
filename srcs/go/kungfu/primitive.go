package kungfu

import (
	"sync"

	"github.com/luomai/kungfu/srcs/go/algo"
	rch "github.com/luomai/kungfu/srcs/go/rchannel"
	"github.com/luomai/kungfu/srcs/go/wire"
)

func (kf *Kungfu) bcast(buffer []byte, count int, dtype wire.KungFu_Datatype, root int, name string) error {
	n := count * dtype.Size()
	myRank := kf.cluster.MyRank()
	if myRank == root {
		var wg sync.WaitGroup
		for i, task := range kf.cluster.AllPeers() {
			if i != root {
				wg.Add(1)
				func(addr rch.NetAddr) {
					m := rch.NewMessage(buffer[:n])
					kf.router.Send(addr.WithName(name), *m)
					wg.Done()
				}(task.NetAddr)
			}
		}
		wg.Wait()
	} else {
		var m rch.Message
		task := kf.cluster.GetPeer(root)
		kf.router.Recv(task.NetAddr.WithName(name), &m)
		if int(m.Length) != n {
			panic("unexpected recv length")
		}
		copy(buffer[:n], m.Data)
	}
	// TODO: check error
	return nil
}

func (kf *Kungfu) reduce(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, root int, name string) error {
	n := count * dtype.Size()
	copy(recvBuf[:n], sendBuf[:n])
	myRank := kf.cluster.MyRank()
	if myRank == root {
		var lock sync.Mutex
		var wg sync.WaitGroup
		for i, task := range kf.cluster.AllPeers() {
			if i != root {
				wg.Add(1)
				func(addr rch.NetAddr) {
					var m rch.Message
					kf.router.Recv(addr.WithName(name), &m)
					if int(m.Length) != n {
						// FIXME: don't panic
						panic("unexpected recv length")
					}
					lock.Lock()
					algo.AddBy(recvBuf[:n], m.Data, count, dtype, op)
					lock.Unlock()
					wg.Done()
				}(task.NetAddr)
			}
		}
		wg.Wait()
	} else {
		task := kf.cluster.GetPeer(root)
		m := rch.NewMessage(sendBuf[:n])
		kf.router.Send(task.NetAddr.WithName(name), *m)
	}
	// TODO: check error
	return nil
}
