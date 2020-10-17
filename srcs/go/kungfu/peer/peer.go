package peer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/env"
	"github.com/lsds/KungFu/srcs/go/kungfu/execution"
	"github.com/lsds/KungFu/srcs/go/kungfu/runner"
	"github.com/lsds/KungFu/srcs/go/kungfu/session"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/rchannel/server"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Peer struct {
	sync.Mutex

	// immutable
	configServerURL    string
	initClusterVersion int
	parent             plan.PeerID
	self               plan.PeerID
	strategy           base.Strategy
	single             bool
	router             *router
	server             server.Server
	httpClient         http.Client

	// dynamic
	clusterVersion int
	currentSession *session.Session
	currentCluster *plan.Cluster
	updated        bool

	detached bool
}

func New() (*Peer, error) {
	config, err := env.ParseConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return NewFromConfig(config)
}

func NewFromConfig(cfg *env.Config) (*Peer, error) {
	router := NewRouter(cfg.Self)
	server := server.New(cfg.Self, router, config.UseUnixSock)
	var initClusterVersion int
	if len(cfg.InitClusterVersion) > 0 {
		var err error
		initClusterVersion, err = strconv.Atoi(cfg.InitClusterVersion)
		if err != nil {
			return nil, err
		}
	}
	initCluster := &plan.Cluster{
		Runners: cfg.InitRunners,
		Workers: cfg.InitPeers,
	}
	return &Peer{
		configServerURL:    cfg.ConfigServer,
		parent:             cfg.Parent,
		currentCluster:     initCluster,
		self:               cfg.Self,
		strategy:           cfg.Strategy,
		initClusterVersion: initClusterVersion,
		clusterVersion:     initClusterVersion,
		single:             cfg.Single,
		router:             router,
		server:             server,
	}, nil
}

func (p *Peer) Start() error {
	if !p.single {
		if err := p.server.Start(); err != nil {
			return err
		}
		if config.EnableMonitoring {
			monitoringPort := p.self.Port + 10000
			monitor.StartServer(int(monitoringPort))
			monitorAddr := plan.NetAddr{
				IPv4: p.self.IPv4, // FIXME: use pubAddr
				Port: monitoringPort,
			}
			log.Infof("Kungfu peer %s started, monitoring endpoint http://%s/metrics", p.self, monitorAddr)
		}
	}
	p.Update()
	return nil
}

func (p *Peer) Close() error {
	if !p.single {
		if config.EnableMonitoring {
			monitor.StopServer()
		}
		p.server.Close() // TODO: check error
	}
	return nil
}

func (p *Peer) Detached() bool {
	return p.detached
}

// UID returns an immutable unique ID of this peer
func (p *Peer) UID() uint64 {
	hi := uint64(p.self.IPv4)
	lo := (uint64(p.self.Port) << 16) | uint64(uint16(p.initClusterVersion))
	return (hi<<32 | lo)
}

var errSelfNotInCluster = errors.New("self not in cluster")

func (p *Peer) CurrentSession() *session.Session {
	p.Lock()
	defer p.Unlock()
	if p.currentSession == nil {
		p.updateTo(p.currentCluster.Workers)
	}
	return p.currentSession
}

func (p *Peer) Update() bool {
	p.Lock()
	defer p.Unlock()
	return p.updateTo(p.currentCluster.Workers)
}

func (p *Peer) updateTo(pl plan.PeerList) bool {
	if config.EnableStallDetection {
		name := fmt.Sprintf("updateTo(%s)", pl.DebugString())
		defer utils.InstallStallDetector(name).Stop()
	}
	p.server.SetToken(uint32(p.clusterVersion))
	if p.updated {
		log.Debugf("ignore update")
		return true
	}
	log.Debugf("Kungfu::updateTo v%d of %d peers: %s", p.clusterVersion, len(pl), pl)
	p.router.ResetConnections(pl, uint32(p.clusterVersion))
	sess, exist := session.New(p.strategy, p.self, pl, p.router.client, p.router.Collective)
	if !exist {
		return false
	}
	if err := sess.Barrier(); err != nil {
		utils.ExitErr(fmt.Errorf("barrier failed after newSession: %v", err))
	}
	p.currentSession = sess
	p.updated = true
	return true
}

func (p *Peer) consensus(bs []byte) bool {
	sess := p.CurrentSession()
	ok, err := sess.BytesConsensus(bs, "")
	if err != nil {
		utils.ExitErr(err)
	}
	return ok
}

func (p *Peer) propose(cluster plan.Cluster) (bool, bool) {
	if config.EnableStallDetection {
		name := fmt.Sprintf("propose(%s)", cluster.DebugString())
		defer utils.InstallStallDetector(name).Stop()
	}
	if p.currentCluster.Eq(cluster) {
		log.Debugf("ingore unchanged proposal")
		return false, false
	}
	if digest := cluster.Bytes(); !p.consensus(digest) {
		log.Errorf("diverge proposal detected among %d peers! I proposed %s", len(cluster.Workers), cluster.Workers)
		return false, false
	}
	{
		stage := runner.Stage{
			Version: p.clusterVersion + 1,
			Cluster: cluster,
		}
		var notify execution.PeerFunc = func(ctrl plan.PeerID) error {
			ctx, cancel := context.WithTimeout(context.TODO(), config.WaitRunnerTimeout)
			defer cancel()
			n, err := p.router.Wait(ctx, ctrl)
			if err != nil {
				return err
			}
			if n > 0 {
				log.Warnf("%s is up after pinged %d times", ctrl, n+1)
			}
			return p.router.Send(ctrl.WithName("update"), stage.Encode(), connection.ConnControl, 0)
		}
		if err := notify.Par(cluster.Runners); err != nil {
			utils.ExitErr(err)
		}
	}
	func() {
		p.Lock()
		defer p.Unlock()
		if p.currentCluster.Workers.Disjoint(cluster.Workers) {
			log.Errorf("Full update detected: %s -> %s! State will be lost.", p.currentCluster.DebugString(), cluster.DebugString())
		} else if len(cluster.Workers) > 0 && !p.currentCluster.Workers.Contains(cluster.Workers[0]) {
			log.Errorf("New root can't not be a new worker! State will be lost.")
		}
		p.currentCluster = &cluster
		p.clusterVersion++
		p.updated = false
	}()
	_, keep := cluster.Workers.Rank(p.self)
	return true, !keep
}

func (p *Peer) ResizeCluster(newSize int) (bool, bool, error) {
	if p.currentSession.Rank() == 0 {
		if err := p.ProposeNewSize(newSize); err != nil {
			log.Warnf("Peer::ResizeCluster failed: %v", err)
		}
	}
	return p.ResizeClusterFromURL()
}

func (p *Peer) ResizeClusterFromURL() (bool, bool, error) {
	var cluster *plan.Cluster
	for i := 0; ; i++ {
		var err error
		cluster, err = p.getClusterConfig(p.configServerURL)
		if err != nil {
			log.Errorf("getClusterConfig failed: %v, using current config", err)
			cluster = p.currentCluster
		}
		if digest := cluster.Bytes(); p.consensus(digest) {
			if i > 0 {
				log.Infof("New peer list is consistent after failed %d times", i)
			} else {
				log.Debugf("New peer list is consistent after ONE attempt!")
			}
			break
		}
		log.Warnf("diverge proposal detected among %d peers! I proposed %s", len(p.currentCluster.Workers), cluster.DebugString())
		time.Sleep(50 * time.Millisecond)
	}
	changed, detached := p.propose(*cluster)
	if detached {
		p.detached = true
	} else {
		p.Update()
	}
	return changed, detached, nil
}

func (p *Peer) getClusterConfig(url string) (*plan.Cluster, error) {
	f, err := utils.OpenURL(url, &p.httpClient, fmt.Sprintf("KungFu Peer: %s", p.self))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cluster plan.Cluster
	if err = json.NewDecoder(f).Decode(&cluster); err != nil {
		return nil, err
	}
	return &cluster, nil
}
