package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/fakemodel"
)

var (
	maxStep    = flag.Int("max-step", 10, "")
	runTrain   = flag.Bool("train", true, "")
	errorAfter = flag.Duration("error-after", 20*time.Second, "")
	adaptive   = flag.Bool("adaptive", true, "")
)

func main() {
	flag.Parse()
	kungfu, err := kf.New()
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()
	self := kungfu.Self()
	initRank := kungfu.CurrentSession().Rank()
	if initRank == 0 {
		ch := make(chan bool, 1)
		go func() {
			fakeTrainLoop(kungfu)
			ch <- true
		}()
		select {
		case <-time.After(*errorAfter):
			fmt.Printf("%s rank %d exits at %d\n", self, initRank, time.Now().UnixNano())
			os.Exit(0)
		case <-ch:
			return
		}
	} else if initRank == 1 {
		ch := make(chan bool, 1)
		go func() {
			fakeTrainLoop(kungfu)
			ch <- true
		}()
		select {
		case <-time.After(*errorAfter * 3):
			fmt.Printf("%s rank %d exits at %d\n", self, initRank, time.Now().UnixNano())
			os.Exit(0)
		case <-ch:
			return
		}
	} else {
		fakeTrainLoop(kungfu)
	}
}

func fakeTrainStep(kungfu *kf.Kungfu, m *fakemodel.FakeModel, step int) {
	sess := kungfu.CurrentSession()
	np := sess.ClusterSize()
	rank := sess.Rank()
	t0 := time.Now()
	numErrors := 0
	for _, name := range m.Names {
		b := m.Buffers[name]
		w := kf.Workspace{
			SendBuf: b.SendBuf,
			RecvBuf: b.RecvBuf,
			OP:      kb.SUM,
			Name:    name,
		}
		err := sess.AllReduce(w)
		if err != nil {
			log.Warnf("AllReduce failed: %v", err)
			numErrors = numErrors + 1
		}
	}
	if numErrors > 0 {
		fmt.Printf("DO NOT UPDATE\n")
	}
	fmt.Printf("step: %d, rank=%d, np=%d, took %s, got %d errors\n", step, rank, np, time.Since(t0), numErrors)
	if sess.Rank() == 0 {
		time.Sleep(600 * time.Millisecond)
	} else {
		time.Sleep(100 * time.Millisecond)
	}
}

func fakeTrainLoop(kungfu *kf.Kungfu) {
	model := fakemodel.New([]int{1}, kb.F32, false)

	// BEGIN tf.train.SessionRunHook::begin
	shouldSync := true
	// END tf.train.SessionRunHook::begin

	for step := 0; step < *maxStep; step++ {
		// BEGIN tf.train.SessionRunHook::before_run
		if shouldSync {
			newStep := syncStep(kungfu, step)
			fmt.Printf("sync step: %d -> %d\n", step, newStep)
			step = newStep
			// TODO: broadcast from the oldest
			shouldSync = false
		}
		// END tf.train.SessionRunHook::before_run

		if *runTrain {
			fakeTrainStep(kungfu, model, step)
		}

		// BEGIN tf.train.SessionRunHook::after_run
		if *adaptive {
			changed, keep := resize(kungfu)
			if !keep {
				break
			}
			if changed {
				shouldSync = true
			}
		}
		// BEGIN tf.train.SessionRunHook::after_run
	}
	log.Infof("finished")
}

func syncStep(kungfu *kf.Kungfu, step int) int {
	sess := kungfu.CurrentSession()
	x := kb.NewVector(1, kb.I64)
	y := kb.NewVector(1, kb.I64)
	x.AsI64()[0] = int64(step)
	w := kf.Workspace{
		SendBuf: x,
		RecvBuf: y,
		OP:      kb.MAX,
		Name:    "sync-step",
	}
	sess.AllReduce(w)
	return int(y.AsI64()[0])
}

func resize(kungfu *kf.Kungfu) (bool, bool) {
	sess := kungfu.CurrentSession()
	oldRank := sess.Rank()
	oldSize := sess.ClusterSize()
	t0 := time.Now()
	changed, keep, err := kungfu.ResizeClusterFromURL()
	if err != nil {
		utils.ExitErr(err)
	}
	if changed {
		if keep {
			sess := kungfu.CurrentSession()
			newRank := sess.Rank()
			newSize := sess.ClusterSize()
			log.Infof("resize %d -> %d took %s, rank %d -> %d", oldSize, newSize, time.Since(t0), oldRank, newRank)
			fmt.Printf("RECOVERED AT %d\n", time.Now().UnixNano())
		} else {
			log.Infof("resize took %s, I'm not in the cluster of %d peers any more.", time.Since(t0), oldSize)
		}
	}
	return changed, keep
}
