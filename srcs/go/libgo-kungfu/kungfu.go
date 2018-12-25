package main

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luomai/kungfu/srcs/go/algo"
	"github.com/luomai/kungfu/srcs/go/log"
	"github.com/luomai/kungfu/srcs/go/metrics"
	rch "github.com/luomai/kungfu/srcs/go/rchannel"
	"github.com/luomai/kungfu/srcs/go/wire"
)

// #include <kungfu.h>
// #include <callback.h>
import "C"

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}

var (
	cluster *rch.ClusterSpec
	router  *rch.Router
	server  *rch.Server

	useFullSymmetricAllReduce = true
	useRingAllReduce          = true
)

func exitErr(err error) {
	log.Errorf("exit on error: %v", err)
	os.Exit(1)
}

//export GoKungfuInit
func GoKungfuInit() int {
	log.Infof("GoKungfuInit")
	if useFullSymmetricAllReduce {
		log.Infof("FullSymmetricAllReduce enabled")
	}
	if useRingAllReduce {
		log.Infof("RingAllReduce enabled")
	}
	var err error
	cluster, err = rch.NewClusterSpecFromEnv()
	if err != nil {
		exitErr(err)
	}
	router, err = rch.NewRouter(cluster)
	if err != nil {
		exitErr(err)
	}
	server, err = rch.NewServer(router)
	if err != nil {
		exitErr(err)
	}
	go metrics.ListenAndServe(cluster.Self.MonitoringPort)
	go server.ListenAndServe()
	go func() {
		for range time.Tick(1 * time.Second) {
			router.UpdateRate()
		}
	}()
	return 0
}

//export GoKungfuFinalize
func GoKungfuFinalize() int {
	log.Infof("GoKungfuFinalize")
	server.Close()
	// TODO: check error
	filename := fmt.Sprintf("vars.%02d.json", cluster.MyRank())
	f, err := os.Create(filename)
	if err != nil {
		return 1
	}
	defer f.Close()
	metrics.RecordStop()
	metrics.Export(f)
	return 0
}

func bcast(buffer []byte, count int, dtype C.KungFu_Datatype, root int, name string) error {
	n := count * wire.KungFu_Datatype(dtype).Size()
	myRank := cluster.MyRank()
	if myRank == root {
		var wg sync.WaitGroup
		for i, task := range cluster.Peers {
			if i != root {
				wg.Add(1)
				func(addr rch.NetAddr) {
					m := rch.NewMessage(buffer[:n])
					router.Send(addr.WithName(name), *m)
					wg.Done()
				}(task.NetAddr)
			}
		}
		wg.Wait()
	} else {
		var m rch.Message
		task := cluster.Peers[root]
		router.Recv(task.NetAddr.WithName(name), &m)
		if int(m.Length) != n {
			panic("unexpected recv length")
		}
		copy(buffer[:n], m.Data)
	}
	// TODO: check error
	return nil
}

func reduce(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, root int, name string) error {
	n := count * wire.KungFu_Datatype(dtype).Size()
	copy(recvBuf[:n], sendBuf[:n])
	myRank := cluster.MyRank()
	if myRank == root {
		var lock sync.Mutex
		var wg sync.WaitGroup
		for i, task := range cluster.Peers {
			if i != root {
				wg.Add(1)
				func(addr rch.NetAddr) {
					var m rch.Message
					router.Recv(addr.WithName(name), &m)
					if int(m.Length) != n {
						// FIXME: don't panic
						panic("unexpected recv length")
					}
					lock.Lock()
					algo.AddBy(recvBuf[:n], m.Data, count, wire.KungFu_Datatype(dtype), wire.KungFu_Op(op))
					lock.Unlock()
					wg.Done()
				}(task.NetAddr)
			}
		}
		wg.Wait()
	} else {
		task := cluster.Peers[root]
		m := rch.NewMessage(sendBuf[:n])
		router.Send(task.NetAddr.WithName(name), *m)
	}
	// TODO: check error
	return nil
}

// rootedAllReduceFunc is the signature of allReduce algorithms that has a central root node
type rootedAllReduceFunc func(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, root int, name string) error

func simpleAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, root int, name string) error {
	if err := reduce(sendBuf, recvBuf, count, dtype, op, root, name); err != nil {
		return err
	}
	return bcast(recvBuf, count, dtype, root, name)
}

func circularAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, root int, name string) error {
	n := count * wire.KungFu_Datatype(dtype).Size()

	sendTo := func(rank int) {
		task := cluster.Peers[rank]
		m := rch.NewMessage(sendBuf[:n])
		router.Send(task.NetAddr.WithName(name), *m)
	}

	recvAdd := func(rank int) {
		task := cluster.Peers[rank]
		addr := task.NetAddr
		var m rch.Message
		router.Recv(addr.WithName(name), &m)
		if int(m.Length) != n {
			// FIXME: don't panic
			panic("unexpected recv length")
		}
		algo.AddBy(recvBuf[:n], m.Data, count, wire.KungFu_Datatype(dtype), wire.KungFu_Op(op))
	}

	recvAssign := func(rank int) {
		task := cluster.Peers[rank]
		addr := task.NetAddr
		var m rch.Message
		router.Recv(addr.WithName(name), &m)
		if int(m.Length) != n {
			// FIXME: don't panic
			panic("unexpected recv length")
		}
		copy(recvBuf[:n], m.Data)
	}

	// k := len(cluster.Peers) // k >= 3
	myRank := cluster.MyRank()
	myPrev, myNext := cluster.PrevAndNext(myRank)
	rootPrev, rootNext := cluster.PrevAndNext(root)

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
			// RSr
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

// symmetricAllReduce parts the original data into k parts and apply an rooted allReduce stratrgy to each part
func symmetricAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string, allReduce rootedAllReduceFunc) error {
	k := len(cluster.Peers)
	blockCount := ceilDiv(count, k)
	blockSize := blockCount * wire.KungFu_Datatype(dtype).Size()

	var wg sync.WaitGroup
	var failed int32
	for i := 0; i < k; i++ {
		wg.Add(1)
		go func(i int) {
			offset := i * blockSize
			currentBlockCount := minInt(blockCount, count-blockCount*i)
			fullName := name + fmt.Sprintf(":part=%d", i) // TODO: use tag
			if err := allReduce(sendBuf[offset:], recvBuf[offset:], currentBlockCount, dtype, op, i, fullName); err != nil {
				log.Warnf("part %d failed: %v", i, err)
				atomic.AddInt32(&failed, 1)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	if failed > 0 {
		return fmt.Errorf("%d parts failed", failed)
	}
	return nil
}

func fullSymmetricAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string) error {
	return symmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name, simpleAllReduce)
}

func ringAllReduce(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string) error {
	k := len(cluster.Peers)
	if k < 3 {
		return fmt.Errorf("ringAllReduce requires k >= 3, but k=%d", k)
	}
	return symmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name, circularAllReduce)
}

//export GoKungfuNegotiate
func GoKungfuNegotiate(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string) int {
	k := len(cluster.Peers)
	if useRingAllReduce {
		if count >= k && k >= 3 {
			return code(ringAllReduce(sendBuf, recvBuf, count, dtype, op, name))
		}
	}

	if useFullSymmetricAllReduce {
		if count >= k {
			return code(fullSymmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name))
		}
		log.Warnf("data size (%d) is smaller that cluster size, will not use fullSymmetricAllReduce", count, k)
	}
	const defaultRoot = 0
	return code(simpleAllReduce(sendBuf, recvBuf, count, dtype, op, defaultRoot, name))
}

//export GoKungfuNegotiateAsync
func GoKungfuNegotiateAsync(sendBuf []byte, recvBuf []byte, count int, dtype C.KungFu_Datatype, op C.KungFu_Op, name string, done *C.callback_t) int {
	name = string([]byte(name)) // TODO: verify that name is cloned
	go func() {
		GoKungfuNegotiate(sendBuf, recvBuf, count, dtype, op, name)
		if done != nil {
			C.invoke_callback(done)
			C.delete_callback(done)
		} else {
			log.Warnf("done is nil!")
		}
	}()
	return 0
}
