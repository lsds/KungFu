package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
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
	algo := kb.ParseAlgo(os.Getenv(kb.AllReduceAlgoEnvKey))
	config := kf.Config{Algo: algo}
	kungfu, err := kf.New(config)
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()

	model := fakemodel.New(fakemodel.Models[*model], kb.KungFu_FLOAT, false)
	fakeTrain(kungfu, model)
}

func getClusterSize(kungfu *kf.Kungfu) int {
	sess := kungfu.CurrentSession()
	return sess.ClusterSize()
}

func logEstimatedSpeed(batches int, batchSize int, d time.Duration, np int) {
	imgPerSec := float64(batches*batchSize) / (float64(d) / float64(time.Second))
	fmt.Fprintf(os.Stderr, "Img/sec %.2f per worker, Img/sec %.2f per cluster, np=%d\n",
		imgPerSec, imgPerSec*float64(np), np)
}

func fakeTrain(kungfu *kf.Kungfu, model *fakemodel.FakeModel) {
	var step int
	t0 := time.Now()
	for i := 0; i < *nIters; i++ {
		for j := 0; j < *stepPerIter; j++ {
			step++
			trainStep(kungfu, model)
		}
		fmt.Printf("after %d steps\n", step)
	}
	np := getClusterSize(kungfu)
	logEstimatedSpeed(*nIters**stepPerIter, *batchSize,
		time.Since(t0), np)
}

func trainStep(kungfu *kf.Kungfu, m *fakemodel.FakeModel) {
	for _, name := range m.Names {
		b := m.Buffers[name]
		w := kf.Workspace{
			SendBuf: b.SendBuf,
			RecvBuf: b.RecvBuf,
			OP:      kb.KungFu_SUM,
			Name:    name,
		}
		sess := kungfu.CurrentSession()
		sess.AllReduce(w)
	}
}
