package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	maxStep  = flag.Int("max-step", 10, "")
	runTrain = flag.Bool("train", true, "")
)

func main() {
	flag.Parse()
	strategy, err := kb.ParseStrategy(os.Getenv(kb.AllReduceStrategyEnvKey))
	if err != nil {
		utils.ExitErr(err)
	}
	config := kf.Config{Strategy: *strategy}
	kungfu, err := kf.New(config)
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()

	fakeTrain(kungfu)
}

func fakeTrain(kungfu *kf.Kungfu) {
	x := kb.NewVector(1, kb.I32)
	y := kb.NewVector(1, kb.I32)
	x.AsI32()[0] = 1
	y.AsI32()[0] = 0

	var step int
	if err := restore(kungfu, &step); err != nil {
		utils.ExitErr(err)
	}

	for ; step < *maxStep; step++ {
		train := func() {
			t0 := time.Now()
			w := kf.Workspace{
				SendBuf: x,
				RecvBuf: y,
				OP:      kb.SUM,
				Name:    "",
			}
			sess := kungfu.CurrentSession()
			np := sess.ClusterSize()
			rank := sess.Rank()
			sess.AllReduce(w)
			fmt.Printf("step: %d, result: %d, rank=%d, np=%d, took %s\n", step, y.AsI32()[0], rank, np, time.Since(t0))
		}
		if *runTrain {
			train()
		}

		if nextStep := step + 1; nextStep < *maxStep && !resize(kungfu, nextStep) {
			log.Infof("should stop")
			break
		}
	}
	log.Infof("finished")
}

func restore(kungfu *kf.Kungfu, step *int) error {
	ckpt := kungfu.GetCheckpoint()
	n, err := strconv.Atoi(ckpt)
	if err != nil {
		return err
	}
	*step = n
	log.Infof("restored from %q", ckpt)
	return nil
}

func resize(kungfu *kf.Kungfu, nextStep int) bool {
	sess := kungfu.CurrentSession()
	np := sess.ClusterSize()
	newSize := np + 1
	t0 := time.Now()
	keep, err := kungfu.ResizeCluster(strconv.Itoa(nextStep), newSize)
	if err != nil {
		utils.ExitErr(err)
	}
	if np != newSize {
		log.Infof("resize %d -> %d took %s", np, newSize, time.Since(t0))
	}
	return keep
}
