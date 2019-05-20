package main

import (
	"errors"
	"flag"
	"log"
	"os"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func main() {
	flag.Parse()
	log.Printf("[%s]", os.Args[0])
	algo := kb.ParseAlgo(os.Getenv(kb.AllReduceAlgoEnvKey))
	config := kf.Config{Algo: algo}
	kungfu, err := kf.New(config)
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer func() {
		time.Sleep(2 * time.Second)
		kungfu.Close() // FIXME: make it graceful
	}()
	sess := kungfu.CurrentSession()
	np := sess.ClusterSize()
	if np < 2 {
		utils.ExitErr(errors.New("cluster_size >= 2 is required"))
	}
	rank := sess.Rank()

	b := newFakeBuffer("test-buffer", 10)

	// collective example
	sess.AllReduce(kf.Workspace{
		SendBuf: b.sendBuf,
		RecvBuf: b.recvBuf,
		OP:      kb.KungFu_SUM,
		Name:    b.name,
	})

	// p2p example
	switch rank {
	case 0:
		sess.SendTo(1, kf.Workspace{
			SendBuf: b.sendBuf,
			Name:    b.name,
		})
	default:
	}
}

type fakeBuffer struct {
	sendBuf *kb.Buffer
	recvBuf *kb.Buffer
	name    string
}

func newFakeBuffer(name string, size int) *fakeBuffer {
	const dSize = 4
	dType := kb.KungFu_FLOAT
	return &fakeBuffer{
		sendBuf: &kb.Buffer{
			Data:  make([]byte, size*dSize),
			Count: size,
			Type:  dType,
		},
		recvBuf: &kb.Buffer{
			Data:  make([]byte, size*dSize),
			Count: size,
			Type:  dType,
		},
		name: name,
	}
}
