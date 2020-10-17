package main

import (
	"flag"
	"fmt"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
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
	peer, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	peer.Start()
	defer peer.Close()
	fakeTrainLoop(peer)
}

func fakeTrainStep(peer *peer.Peer, m *fakemodel.FakeModel, step int) {
	sess := peer.CurrentSession()
	np := sess.Size()
	rank := sess.Rank()
	t0 := time.Now()
	for _, name := range m.Names {
		b := m.Buffers[name]
		w := kb.Workspace{
			SendBuf: b.SendBuf,
			RecvBuf: b.RecvBuf,
			OP:      kb.SUM,
			Name:    name,
		}
		sess.AllReduce(w)
	}
	fmt.Printf("step: %d, rank=%d, np=%d, took %s\n", step, rank, np, time.Since(t0))
	if sess.Rank() == 0 {
		time.Sleep(600 * time.Millisecond)
	} else {
		time.Sleep(100 * time.Millisecond)
	}
}

func fakeTrainLoop(peer *peer.Peer) {
	model := fakemodel.New([]int{1}, kb.F32, false)

	// BEGIN tf.train.SessionRunHook::begin
	shouldSync := true
	// END tf.train.SessionRunHook::begin

	for step := 0; step < *maxStep; step++ {
		// BEGIN tf.train.SessionRunHook::before_run
		if shouldSync {
			newStep := syncStep(peer, step)
			fmt.Printf("sync step: %d -> %d\n", step, newStep)
			step = newStep
			// TODO: broadcast from the oldest
			shouldSync = false
		}
		// END tf.train.SessionRunHook::before_run

		if *runTrain {
			fakeTrainStep(peer, model, step)
		}

		// BEGIN tf.train.SessionRunHook::after_run
		changed, detached := resize(peer)
		if detached {
			break
		}
		if changed {
			shouldSync = true
		}
		// BEGIN tf.train.SessionRunHook::after_run
	}
	log.Infof("finished")
}

func syncStep(peer *peer.Peer, step int) int {
	sess := peer.CurrentSession()
	x := kb.NewVector(1, kb.I64)
	y := kb.NewVector(1, kb.I64)
	x.AsI64()[0] = int64(step)
	w := kb.Workspace{
		SendBuf: x,
		RecvBuf: y,
		OP:      kb.MAX,
		Name:    "sync-step",
	}
	sess.AllReduce(w)
	return int(y.AsI64()[0])
}

func resize(peer *peer.Peer) (bool, bool) {
	sess := peer.CurrentSession()
	oldRank := sess.Rank()
	oldSize := sess.Size()
	t0 := time.Now()
	changed, detached, err := peer.ResizeClusterFromURL()
	if err != nil {
		utils.ExitErr(err)
	}
	if changed {
		if detached {
			log.Infof("resize took %s, I'm not in the cluster of %d peers any more.", time.Since(t0), oldSize)
		} else {
			sess := peer.CurrentSession()
			newRank := sess.Rank()
			newSize := sess.Size()
			log.Infof("resize %d -> %d took %s, rank %d -> %d", oldSize, newSize, time.Since(t0), oldRank, newRank)
		}
	}
	return changed, detached
}
