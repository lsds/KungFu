package runner

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/proc"
	"github.com/lsds/KungFu/srcs/go/rchannel/server"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/local"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

type watcher struct {
	server  server.Server
	parent  plan.PeerID
	parents plan.PeerList

	job job.Job

	ctx     context.Context
	cancel  context.CancelFunc
	ch      chan Stage
	stopped chan plan.PeerID
	keep    bool

	current plan.Cluster
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
	if w.current.Workers.Disjoint(s.Cluster.Workers) {
		log.Errorf("full update detected: %s -> %s", w.current.DebugString(), s.Cluster.DebugString())
	}
	a, b := w.current.Workers.Diff(s.Cluster.Workers)
	del := a.On(w.parent.IPv4)
	add := b.On(w.parent.IPv4)
	log.Infof("arrived at v%d, new np=%d, local: +%d/-%d, global: +%d/-%d", s.Version, len(s.Cluster.Workers), len(add), len(del), len(b), len(a))
	log.Debugf("waiting %d peers to stop", len(del))
	for _, id := range del {
		w.delete(id)
	}
	log.Debugf("%s removed: %d - %d = %d", utils.Pluralize(len(del), "peer", "peers"), len(w.current.Workers), len(del), len(w.current.Workers)-len(del))
	for _, id := range add {
		w.create(id, s)
	}
	log.Debugf("%s created: %d - %d + %d = %d", utils.Pluralize(len(add), "peer", "peers"), len(w.current.Workers), len(del), len(add), len(s.Cluster.Workers))
	w.current = s.Cluster
}

func (w *watcher) watchRun(globalCtx context.Context) {
	for {
		select {
		case s := <-w.ch:
			w.update(s)
		case <-w.stopped:
			n := atomic.AddInt32(&w.running, -1)
			log.Debugf("%s are still running on this host", utils.Pluralize(int(n), "peer", "peers"))
			if n == 0 && !w.keep {
				return
			}
		case <-w.ctx.Done():
			log.Errorf("canceled: %v", w.ctx.Err())
			return
		case <-globalCtx.Done():
			log.Errorf("canceled: %v", globalCtx.Err())
			return
		}
	}
}

func WatchRun(ctx context.Context, self plan.PeerID, runners plan.PeerList, ch chan Stage, j job.Job, keep bool, debugPort int) {
	ctx, cancel := context.WithCancel(ctx)
	globalCtx, globalCancel := context.WithCancel(ctx)
	handler := NewHandler(self, ch, globalCancel)
	if debugPort > 0 {
		log.Infof("debug server: http://127.0.0.1:%d/", debugPort)
		go http.ListenAndServe(net.JoinHostPort("", strconv.Itoa(debugPort)), handler)
	}
	server := server.New(self, handler, config.UseUnixSock)
	if err := server.Start(); err != nil {
		utils.ExitErr(err)
	}
	defer server.Close()
	watcher := &watcher{
		server:  server,
		parent:  self,
		parents: runners,
		job:     j,
		ctx:     ctx,
		cancel:  cancel,
		ch:      ch,
		keep:    keep,
		stopped: make(chan plan.PeerID, 1),
		gs:      make(map[plan.PeerID]*sync.WaitGroup),
		gpuPool: job.NewGPUPool(j.HostList.SlotOf(self.IPv4)),
	}
	log.Infof("watching config server")
	watcher.watchRun(globalCtx)
	log.Infof(xterm.Blue.S("stop watching"))
}

func runProc(ctx context.Context, cancel context.CancelFunc, p proc.Proc, version int, logDir string) {
	r := &local.Runner{
		Name:          p.Name,
		LogDir:        logDir,
		LogFilePrefix: fmt.Sprintf("%s@%d", p.Name, version),
		VerboseLog:    true,
	}
	if err := r.TryRun(ctx, p); err != nil {
		log.Infof("%s finished with error: %v", p.Name, err)
		cancel()
		utils.ExitErr(err) // FIXME: graceful shutdown
		return
	}
}
