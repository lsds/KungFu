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
	"github.com/lsds/KungFu/tests/go/fakemodel"
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

	fakeTrainLoop(kungfu)
}

func fakeTrainStep(kungfu *kf.Kungfu, m *fakemodel.FakeModel, step int) {
	sess := kungfu.CurrentSession()
	np := sess.ClusterSize()
	rank := sess.Rank()
	t0 := time.Now()
	for _, name := range m.Names {
		b := m.Buffers[name]
		w := kf.Workspace{
			SendBuf: b.SendBuf,
			RecvBuf: b.RecvBuf,
			OP:      kb.SUM,
			Name:    name,
		}
		sess.AllReduce(w)
	}
	fmt.Printf("step: %d, rank=%d, np=%d, took %s\n", step, rank, np, time.Since(t0))
	if sess.Rank() == 0 {
		time.Sleep(2000 * time.Millisecond)
	} else {
		time.Sleep(100 * time.Millisecond)
	}
}

func fakeTrainLoop(kungfu *kf.Kungfu) {
	model := fakemodel.New([]int{1}, kb.F32, false)
	var step int
	if err := restore(kungfu, &step); err != nil {
		utils.ExitErr(err)
	}
	for ; step < *maxStep; step++ {
		if *runTrain {
			fakeTrainStep(kungfu, model, step)
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
	n, err := strconv.Atoi(initStep)
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
