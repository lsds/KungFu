package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/fakemodel"
)

var (
	batchSize   = flag.Int("batch-size", 32, "")
	imgPerSec   = flag.Int("img-per-sec", 185, "")
	nIters      = flag.Int("n-iters", 11, "")
	stepPerIter = flag.Int("step-per-iter", 10, "")
	model       = flag.String("model", fakemodel.Names[0], strings.Join(fakemodel.Names, " | "))
)

func main() {
	flag.Parse()
	log.Printf("[%s]", os.Args[0])
	peer, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	peer.Start()
	defer peer.Close()

	model := fakemodel.New(fakemodel.Models[*model], kb.F32, false)
	fakeTrain(peer, model)
}

func getClusterSize(peer *peer.Peer) int {
	sess := peer.CurrentSession()
	return sess.Size()
}

func logEstimatedSpeed(batches int, batchSize int, d time.Duration, np int) {
	imgPerSec := float64(batches*batchSize) / (float64(d) / float64(time.Second))
	fmt.Fprintf(os.Stderr, "Img/sec %.2f per worker, Img/sec %.2f per cluster, np=%d\n",
		imgPerSec, imgPerSec*float64(np), np)
}

func fakeTrain(peer *peer.Peer, model *fakemodel.FakeModel) {
	var step int
	t0 := time.Now()
	for i := 0; i < *nIters; i++ {
		for j := 0; j < *stepPerIter; j++ {
			step++
			trainStep(peer, model)
		}
		fmt.Printf("after %d steps\n", step)
	}
	np := getClusterSize(peer)
	logEstimatedSpeed(*nIters**stepPerIter, *batchSize,
		time.Since(t0), np)
}

func trainStep(peer *peer.Peer, m *fakemodel.FakeModel) {
	for _, name := range m.Names {
		b := m.Buffers[name]
		w := kb.Workspace{
			SendBuf: b.SendBuf,
			RecvBuf: b.RecvBuf,
			OP:      kb.SUM,
			Name:    name,
		}
		sess := peer.CurrentSession()
		sess.AllReduce(w)
	}
}
