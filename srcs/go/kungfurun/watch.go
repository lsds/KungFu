package kungfurun

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/lsds/KungFu/srcs/go/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/utils"
	runner "github.com/lsds/KungFu/srcs/go/utils/runner/local"
)

type watcher struct {
	server  rch.Server
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
	gpuPool *job.GPUPool
}

func (w *watcher) create(id plan.PeerID, s Stage) {
	w.gs[id] = new(sync.WaitGroup)
	w.gs[id].Add(1)
	atomic.AddInt32(&w.running, 1)
	gpuID := w.gpuPool.Get()
	if gpuID < 0 {
		log.Errorf("gpuID = %d", gpuID)
	}
	proc := w.job.NewProc(id, gpuID, s.Version, s.Cluster)
	go func(g *sync.WaitGroup) {
		runProc(w.ctx, w.cancel, proc, s.Version, w.job.LogDir)
		g.Done()
		w.gpuPool.Put(gpuID)
		w.stopped <- id
	}(w.gs[id])
}

func (w *watcher) delete(id plan.PeerID) {
	w.gs[id].Wait()
	delete(w.gs, id)
}

func (w *watcher) update(s Stage) {
	w.server.SetToken(uint32(s.Version))
	if w.current.Disjoint(s.Cluster) {
		log.Warnf("full update detected: %s -> %s", w.current.DebugString(), s.Cluster.DebugString())
	}
	a, b := w.current.Diff(s.Cluster)
	del := a.On(w.parent.IPv4)
	add := b.On(w.parent.IPv4)
	log.Infof("arrived at v%d, np=%d, +%d/%d, -%d/%d", s.Version, len(s.Cluster), len(add), len(b), len(del), len(a))
	log.Debugf("waiting %d peers to stop", len(del))
	for _, id := range del {
		w.delete(id)
	}
	log.Debugf("%s removed: %d - %d = %d", utils.Pluralize(len(del), "peer", "peers"), len(w.current), len(del), len(w.current)-len(del))
	for _, id := range add {
		w.create(id, s)
	}
	log.Debugf("%s created: %d - %d + %d = %d", utils.Pluralize(len(add), "peer", "peers"), len(w.current), len(del), len(add), len(s.Cluster))
	w.current = s.Cluster
}

func (w *watcher) watchRun(globalCtx context.Context) {
	hostRank, _ := w.parents.Rank(w.parent)
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
			log.Debugf("%s are still running on this host", utils.Pluralize(int(n), "peer", "peers"))
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

func WatchRun(ctx context.Context, parent plan.PeerID, parents plan.PeerList, ch chan Stage, j job.Job, debugPort int) {
	ctx, cancel := context.WithCancel(ctx)
	globalCtx, globalCancel := context.WithCancel(ctx)
	handler := NewHandler(parent, ch, globalCancel)
	if debugPort > 0 {
		log.Infof("debug server: http://127.0.0.1:%d/", debugPort)
		go http.ListenAndServe(net.JoinHostPort("", strconv.Itoa(debugPort)), handler)
	}
	server := rch.NewServer(handler)
	if err := server.Start(); err != nil {
		utils.ExitErr(err)
	}
	defer server.Close()
	watcher := &watcher{
		server:  server,
		parent:  parent,
		parents: parents,

		job:     j,
		ctx:     ctx,
		cancel:  cancel,
		ch:      ch,
		stopped: make(chan plan.PeerID, 1),
		gs:      make(map[plan.PeerID]*sync.WaitGroup),
		gpuPool: job.NewGPUPool(j.HostList.SlotOf(parent.IPv4)),
	}
	log.Infof("watching config server")
	watcher.watchRun(globalCtx)
	log.Infof("stop watching")
}

func runProc(ctx context.Context, cancel context.CancelFunc, proc job.Proc, version int, logDir string) {
	r := &runner.Runner{
		Name:          proc.Name,
		LogDir:        logDir,
		LogFilePrefix: fmt.Sprintf("%s@%d", proc.Name, version),
		VerboseLog:    true,
	}
	if err := r.Run(ctx, proc.Cmd()); err != nil {
		log.Infof("%s finished with error: %v", proc.Name, err)
		cancel()
		utils.ExitErr(err) // FIXME: graceful shutdown
		return
	}
}
