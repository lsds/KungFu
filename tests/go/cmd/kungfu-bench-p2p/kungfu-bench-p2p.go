package main

import (
	"flag"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
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
	log.SetFlags(0)
	flag.Parse()
	peer, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	peer.Start()
	defer peer.Close()
	sizes, ok := fakemodel.Models[*model]
	if !ok {
		log.Exitf("invalid model name: %s", *model)
	}
	m := fakemodel.New(sizes, kb.F32, *fuse)
	benchPeerToPeer(peer, m)
}

func benchPeerToPeer(peer *peer.Peer, m *fakemodel.FakeModel) {
	sess := peer.CurrentSession()
	rank := sess.Rank()

	if rank == 0 {
		m.ShowInfo()
		log.Infof("mode: %s", *mode)
	}

	sess.Barrier()
	defer sess.Barrier()

	for _, name := range m.Names {
		b := m.Buffers[name]
		peer.Save(name, b.SendBuf)
	}

	np := sess.Size()
	target := rank
	s := &selector{n: np}
	if np > 1 {
		target = s.NextExclude(rank)
	}

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
				peer.RequestRank(target, "", w.Name, w.RecvBuf)
			})
		}(name, m.Buffers[name])
	}
	fns := map[string]func(){
		"par": g.Par,
		"seq": g.Seq,
	}
	runEpoch, ok := fns[*mode]
	if !ok {
		log.Exitf("invalid mode: %s", *mode)
	}

	multiplier := 2 * np
	workload := int64(*epochs) * int64(multiplier) * int64(m.Size())

	var duration time.Duration
	logRate := func(d time.Duration) {
		duration = d
		log.Infof("took %s, rate: %s\n", d, testutils.ShowRate(workload, duration))
	}

	for i := 0; i < *warmupEpochs; i++ {
		if rank == 0 {
			log.Infof("warmup epoch %d", i+1)
		}
		runEpoch()
		sess.Barrier()
	}

	func() {
		defer testutils.NewStopWatch().Stop(logRate)
		for i := 0; i < *epochs; i++ {
			if rank == 0 {
				log.Infof("epoch %d", i+1)
			}
			runEpoch()
			sess.Barrier()
		}
	}()

	if rank == 0 {
		log.Infof("Result: model: %s, %s, mode: %s, rate: %s", *model, m.Info(), *mode, testutils.ShowRate(workload, duration))
	}
}

type selector struct {
	k int
	n int
}

func (s *selector) Next() int {
	s.k = (s.k + 1) % s.n
	return s.k
}

func (s *selector) NextExclude(self int) int {
	for {
		k := s.Next()
		if k != self {
			return k
		}
	}
}
