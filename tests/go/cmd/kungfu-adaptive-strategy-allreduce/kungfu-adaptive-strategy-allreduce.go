package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/kungfu/session"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/tests/go/fakemodel"
	"github.com/lsds/KungFu/tests/go/taskgroup"
	"github.com/lsds/KungFu/tests/go/testutils"
)

var (
	model          = flag.String("model", fakemodel.Names[0], strings.Join(fakemodel.Names, " | "))
	mode           = flag.String("mode", "seq", "par | seq")
	fuse           = flag.Bool("fuse", false, "")
	epochs         = flag.Int("epochs", 15, "")
	warmupEpochs   = flag.Int("warmup", 2, "warmup epochs")
	stats          []session.StrategyStat
	strategyOffset = 1
)

func main() {
	flag.Parse()
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
	m := fakemodel.New(sizes, kb.I32, *fuse)
	for _, db := range m.Buffers {
		x := db.SendBuf.AsI32()
		for i := range x {
			x[i] = 1
		}
	}

	for _, db := range m.Buffers {
		x := db.SendBuf.AsI32()
		for i := range x {
			if x[i] != 1 {
				panic("Value assignment failed!")
			}
		}
	}

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

	np := sess.Size()

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
				//truncate Rcv buffer
				y := b.RecvBuf.AsI32()
				for i := range y {
					y[i] = 0
				}

				err := sess.SmartAllReduce(w)
				if err != nil {
					utils.ExitErr(fmt.Errorf("%s failed performing allreduce", `kungfu-adaptive-strategy-allreduce`))
				}

				//validate received data
				for i := range y {
					if y[i] != int32(np) {
						utils.ExitErr(fmt.Errorf("%s failed data validation", `kungfu-adaptive-strategy-allreduce`))
					}
				}
			})
		}(name, m.Buffers[name])
	}

	multiplier := 4 * (np - 1)
	workload := int64(*epochs) * int64(multiplier) * int64(m.Size())

	var duration time.Duration
	// logRate := func(d time.Duration) {
	// 	duration = d
	// 	log.Infof("took %s, rate: %s\n", d, testutils.ShowRate(workload, duration))
	// }

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
		var monTasks taskgroup.Group

		db := fakemodel.NewDoubleBuffer(kb.I8, sess.GetNumStrategies())

		monTasks.Add(func() {

			w := kb.Workspace{
				SendBuf: db.SendBuf,
				RecvBuf: db.RecvBuf,
				OP:      kb.SUM,
				Name:    "StratMon",
			}

			err := sess.AllReduce(w)
			if err != nil {
				utils.ExitErr(fmt.Errorf("%s failed performing allreduce", `kungfu-adaptive-strategy-allreduce`))
			}
		})

		// defer testutils.NewStopWatch().Stop(logRate)
		for i := 0; i < *epochs; i++ {
			if rank == 0 {
				log.Infof("epoch %d", i+1)
			}
			runEpoch()
			stats = append(stats, sess.LogStats(0))

			sess.MonitorStrategy(db.SendBuf.AsI8())

			//else perform AllReduce to reach concensus on chaning changing strategies
			fmt.Println("DEBUG:: about to synch strategies mon")
			monTasks.Seq()
			fmt.Println("DEBUG:: monitoring synced")
			sess.ChangeStrategy(db.RecvBuf.AsI8(), strategyOffset)
		}
	}()

	if rank == 0 {
		log.Infof("Result: model: %s, %s, mode: %s, rate: %s", *model, m.Info(), *mode, testutils.ShowRate(workload, duration))

		sess.PrintSessionState()

		for i, ss := range stats {
			fmt.Println("epoch #", i, ": Master[", 0, "] avgDuration=", ss.AvgDuration, " CMA=", ss.CmaDuration)
		}
	}
}
