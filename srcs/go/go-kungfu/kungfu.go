package main

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/luomai/kungfu/srcs/go/algo"
	"github.com/luomai/kungfu/srcs/go/log"
	"github.com/luomai/kungfu/srcs/go/metrics"
	rch "github.com/luomai/kungfu/srcs/go/rchannel"
	"github.com/luomai/kungfu/srcs/go/wire"
)

// #include <mpi.h>
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
)

func exitErr(err error) {
	log.Errorf("exit on error: %v", err)
	os.Exit(1)
}

//export GoKungfuInit
func GoKungfuInit() int {
	log.Infof("GoKungfuInit")
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

func bcast(buffer []byte, count int, dtype C.MPI_Datatype, root int, name string) int {
	n := count * wire.MPI_Datatype(dtype).Size()
	var wg sync.WaitGroup
	myRank := cluster.MyRank()
	if myRank == root {
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
	} else {
		var m rch.Message
		task := cluster.Peers[root]
		router.Recv(task.NetAddr.WithName(name), &m)
		if int(m.Length) != n {
			panic("unexpected recv length")
		}
		copy(buffer[:n], m.Data)
	}
	wg.Wait()
	// TODO: check error
	return 0
}

//export GoKungfuNegotiate
func GoKungfuNegotiate(sendBuf []byte, recvBuf []byte, count int, dtype C.MPI_Datatype, op C.MPI_Op, name string) int {
	// log.Infof("GoKungfuNegotiate: %s, %d, %d", name, count, dtype)
	root := 0
	n := count * wire.MPI_Datatype(dtype).Size()

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
					buf := m.Data
					lock.Lock()
					algo.AddBy(recvBuf[:n], buf, count, wire.MPI_Datatype(dtype), wire.MPI_Op(op))
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
	return bcast(recvBuf, count, dtype, root, name)
}

//export GoKungfuNegotiateAsync
func GoKungfuNegotiateAsync(sendBuf []byte, recvBuf []byte, count int, dtype C.MPI_Datatype, op C.MPI_Op, name string, done *C.callback_t) int {
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
