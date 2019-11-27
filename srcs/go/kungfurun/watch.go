package kungfurun

import (
	"context"
	"path"
	"sync"
	"sync/atomic"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type watcher struct {
	parent  plan.PeerID
	parents plan.PeerList

	job job.Job

	ctx     context.Context
	cancel  context.CancelFunc
	ch      chan Stage
	stopped chan plan.PeerID

	current plan.PeerList
	running int32
	gs      map[plan.PeerID]*sync.WaitGroup
}

func (w *watcher) create(id plan.PeerID, s Stage) {
	w.gs[id] = new(sync.WaitGroup)
	w.gs[id].Add(1)
	atomic.AddInt32(&w.running, 1)
	localRank, _ := s.Cluster.LocalRank(id)
	proc := w.job.NewProc(id, localRank, s.Checkpoint, s.Cluster)
	go func(g *sync.WaitGroup) {
		runProc(w.ctx, w.cancel, proc, s.Checkpoint, w.job.LogDir)
		g.Done()
		w.stopped <- id
	}(w.gs[id])
}

func (w *watcher) delete(id plan.PeerID) {
	w.gs[id].Wait()
	delete(w.gs, id)
}

func (w *watcher) update(s Stage) {
	a, b := w.current.Diff(s.Cluster)
	del := a.On(w.parent.IPv4)
	add := b.On(w.parent.IPv4)
	log.Infof("arrived at %q, np=%d, +%d/%d, -%d/%d", s.Checkpoint, len(s.Cluster), len(add), len(b), len(del), len(a))
	log.Debugf("waiting %d peers to stop", len(del))
	for _, id := range del {
		w.delete(id)
	}
	log.Debugf("%s removed", utils.Pluralize(len(del), "peer", "peers"))
	for _, id := range add {
		w.create(id, s)
	}
	log.Debugf("%s created", utils.Pluralize(len(add), "peer", "peers"))
	w.current = s.Cluster
}

func (w *watcher) watchRun(globalCtx context.Context) {
	hostRank, _ := w.parents.Lookup(w.parent)
	var globalStopped, inactive bool
	for {
		select {
		case s := <-w.ch:
			w.update(s)
			if n := atomic.LoadInt32(&w.running); n > 0 {
				inactive = false
			}
		case <-w.stopped:
			n := atomic.AddInt32(&w.running, -1)
			log.Debugf("%s is still running on this host", utils.Pluralize(int(n), "peer", "peers"))
			if n == 0 {
				inactive = true
				if hostRank == 0 {
					globalStopped = true
					if len(w.parents) > 0 {
						router := rch.NewRouter(w.parent)
						for i, p := range w.parents {
							if i != hostRank {
								router.Send(p.WithName("exit"), nil, rch.ConnControl, 0)
							}
						}
					}
				}
			}
		case <-globalCtx.Done():
			globalStopped = true
		}
		if globalStopped && inactive {
			break
		}
	}
}

func WatchRun(ctx context.Context, parent plan.PeerID, parents plan.PeerList, ch chan Stage, job job.Job) {
	ctx, cancel := context.WithCancel(ctx)
	watcher := &watcher{
		parent:  parent,
		parents: parents,

		job:     job,
		ctx:     ctx,
		cancel:  cancel,
		ch:      ch,
		stopped: make(chan plan.PeerID, 1),
		gs:      make(map[plan.PeerID]*sync.WaitGroup),
	}

	globalCtx, globalCancel := context.WithCancel(ctx)
	server := rch.NewServer(NewHandler(parent, ch, globalCancel))
	if err := server.Start(); err != nil {
		utils.ExitErr(err)
	}
	defer server.Close()
	log.Infof("watching config server")
	watcher.watchRun(globalCtx)
	log.Infof("stop watching")
}

func runProc(ctx context.Context, cancel context.CancelFunc, proc job.Proc, version string, logDir string) {
	r := &runner.Runner{}
	r.SetName(proc.Name)
	r.SetLogFilePrefix(path.Join(logDir, proc.Name+"@"+version))
	r.SetVerbose(true)
	if err := r.Run(ctx, proc.Cmd()); err != nil {
		log.Infof("%s finished with error: %v", proc.Name, err)
		cancel()
		return
	}
}
