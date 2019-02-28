package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	batchSize   = flag.Int("batch-size", 32, "")
	imgPerSec   = flag.Int("img-per-sec", 185, "")
	nIters      = flag.Int("n-iters", 11, "")
	stepPerIter = flag.Int("step-per-iter", 10, "")
)

type fakeBuffer struct {
	sendBuf *kb.Buffer
	recvBuf *kb.Buffer
	name    string
}

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

	buffers := createFakeBuffers(resnet50GradSizes)
	fakeTrain(kungfu, buffers)
}

func getTestClusterSize() int {
	np, err := strconv.Atoi(os.Getenv(`KUNGFU_TEST_CLUSTER_SIZE`))
	if err != nil {
		utils.ExitErr(err)
	}
	return np
}

func logEstimatedSpeed(batches int, batchSize int, d time.Duration, np int) {
	imgPerSec := float64(batches*batchSize) / (float64(d) / float64(time.Second))
	fmt.Fprintf(os.Stderr, "Img/sec %.2f per worker, Img/sec %.2f per cluster, np=%d\n",
		imgPerSec, imgPerSec*float64(np), np)
}

func fakeTrain(kungfu *kf.Kungfu, buffers []fakeBuffer) {
	np := getTestClusterSize()
	sess := kungfu.CurrentSession()
	var step int
	t0 := time.Now()
	for i := 0; i < *nIters; i++ {
		for j := 0; j < *stepPerIter; j++ {
			step++
			for _, b := range buffers {
				w := kf.Workspace{
					SendBuf: b.sendBuf,
					RecvBuf: b.recvBuf,
					OP:      kb.KungFu_SUM,
					Name:    b.name,
				}
				sess.AllReduce(w)
			}
		}
		fmt.Printf("after %d steps\n", step)
	}
	logEstimatedSpeed(*nIters**stepPerIter, *batchSize,
		time.Since(t0), np)
}

func createFakeBuffers(sizes []int) []fakeBuffer {
	dSize := 4
	dType := kb.KungFu_FLOAT
	var fbs []fakeBuffer
	for i, size := range sizes {
		fb := fakeBuffer{
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
			name: fmt.Sprintf("NegotiatedGrad_%d/AllReduce", i),
		}
		fbs = append(fbs, fb)
	}
	return fbs
}
