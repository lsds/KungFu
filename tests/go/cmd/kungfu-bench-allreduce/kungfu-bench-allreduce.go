package main

import (
	"flag"
	"strings"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	"github.com/lsds/KungFu/srcs/go/kungfu/session"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
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
)

func main() {
	flag.Parse()
	kungfu, err := kf.New()
	if err != nil {
		utils.ExitErr(err)
	}
	kungfu.Start()
	defer kungfu.Close()
	sizes, ok := fakemodel.Models[*model]
	if !ok {
		log.Exitf("invalid model name: %s", *model)
	}
	m := fakemodel.New(sizes, kb.F32, *fuse)
	benchAllReduce(kungfu, m)
}

func benchAllReduce(kungfu *kf.Kungfu, m *fakemodel.FakeModel) {
	sess := kungfu.CurrentSession()
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
				w := session.Workspace{
					SendBuf: b.SendBuf,
					RecvBuf: b.RecvBuf,
					OP:      kb.SUM,
					Name:    name,
				}
				sess.AllReduce(w)
			})
		}(name, m.Buffers[name])
	}

	np := sess.ClusterSize()
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
