package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	batchSize     = flag.Int("batch-size", 32, "")
	imgPerSec     = flag.Int("img-per-sec", 185, "")
	nIters        = flag.Int("n-iters", 11, "")
	stepPerIter   = flag.Int("step-per-iter", 10, "")
	model         = flag.String("model", "resnet50", "resnet50 | mnist-slp")
	enableControl = flag.Bool("control", false, "mock control cluster size")
)

type fakeBuffer struct {
	sendBuf *kb.Buffer
	recvBuf *kb.Buffer
	name    string
}

type fakeModel struct {
	buffers []fakeBuffer
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

	model := &fakeModel{
		buffers: createFakeBuffers(models[*model]),
	}
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

func fakeTrain(kungfu *kf.Kungfu, model *fakeModel) {
	var step int
	t0 := time.Now()
	trainStep := model.trainStep
	if *enableControl {
		trainStep = withControl(trainStep, model.control)
	}
	for i := 0; i < *nIters; i++ {
		for j := 0; j < *stepPerIter; j++ {
			step++
			trainStep(kungfu)
		}
		fmt.Printf("after %d steps\n", step)
	}
	np := getClusterSize(kungfu)
	logEstimatedSpeed(*nIters**stepPerIter, *batchSize,
		time.Since(t0), np)
}

type TrainStep func(kungfu *kf.Kungfu)
type ControlFunc func(kungfu *kf.Kungfu)

func withControl(trainStep TrainStep, controlFunc ControlFunc) TrainStep {
	return func(kungfu *kf.Kungfu) {
		trainStep(kungfu)
		controlFunc(kungfu)
	}
}

func (m *fakeModel) control(kungfu *kf.Kungfu) {
	// TODO: change cluster size
	log.Printf("TODO: control cluster size")
}

func (m *fakeModel) trainStep(kungfu *kf.Kungfu) {
	for _, b := range m.buffers {
		w := kf.Workspace{
			SendBuf: b.sendBuf,
			RecvBuf: b.recvBuf,
			OP:      kb.KungFu_SUM,
			Name:    b.name,
		}
		sess := kungfu.CurrentSession()
		sess.AllReduce(w)
	}
}

func createFakeBuffers(sizes []int) []fakeBuffer {
	const dSize = 4
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
