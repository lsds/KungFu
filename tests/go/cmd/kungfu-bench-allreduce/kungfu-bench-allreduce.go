package main

import (
	"flag"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/nccl"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/fakemodel"
	"github.com/lsds/KungFu/tests/go/taskgroup"
	"github.com/lsds/KungFu/tests/go/testutils"
)

var (
	model        = flag.String("model", fakemodel.Names[0], strings.Join(fakemodel.Names, " | "))
	mode         = flag.String("mode", "seq", "par | seq")
	fuse         = flag.Bool("fuse", false, "")
	epochs       = flag.Int("epochs", 15, "")
	warmupEpochs = flag.Int("warmup", 2, "warmup epochs")

	randomNcclFailure = flag.Bool("rand-nccl-failure", false, "")
)

func main() {
	flag.Parse()
	if *randomNcclFailure {
		nccl.RandomFailure()
	}
	p, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	p.Start()
	defer p.Close()
	sizes, ok := fakemodel.Models[*model]
	if !ok {
		log.Exitf("invalid model name: %s", *model)
	}
	m := fakemodel.New(sizes, kb.F32, *fuse)
	benchAllReduce(p, m)
}

func benchAllReduce(peer *peer.Peer, m *fakemodel.FakeModel) {
	sess := peer.CurrentSession()
	rank := sess.Rank()

	if rank == 0 {
		m.ShowInfo()
		log.Infof("mode: %s", *mode)
	}

	sess.Barrier()

	var g taskgroup.Group
	for _, name := range m.Names {
		func(name string, b fakemodel.DoubleBuffer) {
			g.Add(func() {
				w := kb.Workspace{
					SendBuf: b.SendBuf,
					RecvBuf: b.RecvBuf,
					OP:      kb.SUM,
					Name:    name,
				}
				sess.AllReduce(w)
			})
		}(name, m.Buffers[name])
	}

	np := sess.Size()
	multiplier := 4 * (np - 1)
	workload := int64(*epochs) * int64(multiplier) * int64(m.Size())

	var duration time.Duration
	logRate := func(d time.Duration) {
		duration = d
		log.Infof("took %s, rate: %s\n", d, testutils.ShowRate(workload, duration))
	}

	fns := map[string]func(){
		"par": g.Par,
		"seq": g.Seq,
	}
	runEpoch, ok := fns[*mode]
	if !ok {
		log.Exitf("invalid mode: %s", *mode)
	}

	for i := 0; i < *warmupEpochs; i++ {
		if rank == 0 {
			log.Infof("warmup epoch %d", i+1)
		}
		runEpoch()
	}

	func() {
		defer testutils.NewStopWatch().Stop(logRate)
		for i := 0; i < *epochs; i++ {
			if rank == 0 {
				log.Infof("epoch %d", i+1)
			}
			runEpoch()
		}
	}()

	if rank == 0 {
		log.Infof("Result: model: %s, %s, mode: %s, rate: %s", *model, m.Info(), *mode, testutils.ShowRate(workload, duration))
	}
}
