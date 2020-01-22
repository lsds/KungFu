package main

import (
	"flag"
	"fmt"
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
	kungfu, err := kf.New()
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
			if sess.Rank() == 0 {
				time.Sleep(2000 * time.Millisecond)
			} else {
				time.Sleep(100 * time.Millisecond)
			}
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
	initStep := kungfu.GetInitStep()
	n, err := strconv.Atoi(initStep) // error occurs here
	if err != nil {
		return err
	}
	*step = n
	log.Infof("restored from %q", initStep)
	return nil
}

func resize(kungfu *kf.Kungfu, nextStep int) bool {
	sess := kungfu.CurrentSession()
	npBefore := sess.ClusterSize()
	t0 := time.Now()
	changed, keep, err := kungfu.ResizeClusterFromURL(strconv.Itoa(nextStep))
	if err != nil {
		utils.ExitErr(err)
	}
	if changed {
		npAfter := sess.ClusterSize()
		log.Infof("resize %d -> %d took %s", npBefore, npAfter, time.Since(t0))
	}
	return keep
}
