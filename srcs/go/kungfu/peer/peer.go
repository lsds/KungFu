package peer

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/execution"
	"github.com/lsds/KungFu/srcs/go/kungfu/session"
	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	kc "github.com/lsds/KungFu/srcs/go/kungfu/config"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
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
	strategy           kb.Strategy
	single             bool
	router             *rch.Router
	server             server.Server
	client             http.Client

	// dynamic
	clusterVersion int
	currentSession *session.Session
	currentCluster *plan.Cluster
	updated        bool
}

func New() (*Peer, error) {
	config, err := plan.ParseConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return NewFromConfig(config)
}

func NewFromConfig(config *plan.Config) (*Peer, error) {
	router := rch.NewRouter(config.Self)
	server := server.New(router)
	var initClusterVersion int
	if len(config.InitClusterVersion) > 0 {
		var err error
		initClusterVersion, err = strconv.Atoi(config.InitClusterVersion)
		if err != nil {
			return nil, err
		}
	}
	currentCluster := &plan.Cluster{
		Runners: config.Parents,
		Workers: config.InitPeers,
	}
	return &Peer{
		configServerURL:    config.ConfigServer,
		parent:             config.Parent,
		currentCluster:     currentCluster,
		self:               config.Self,
		strategy:           config.Strategy,
		initClusterVersion: initClusterVersion,
		clusterVersion:     initClusterVersion,
		single:             config.Single,
		router:             router,
		server:             server,
	}, nil
}

func (p *Peer) Start() error {
	if !p.single {
		if err := p.server.Start(); err != nil {
			return err
		}
		if kc.EnableMonitoring {
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
		if kc.EnableMonitoring {
			monitor.StopServer()
		}
		p.server.Close() // TODO: check error
	}
	return nil
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
	if kc.EnableStallDetection {
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
	sess, exist := session.New(p.strategy, p.self, pl, p.router)
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

func (p *Peer) SaveVersion(version, name string, buf *kb.Vector) error {
	return p.router.P2P.SaveVersion(version, name, buf)
}

func (p *Peer) Save(name string, buf *kb.Vector) error {
	return p.router.P2P.Save(name, buf)
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
	if p.currentCluster.Eq(cluster) {
		log.Debugf("ingore unchanged proposal")
		return false, true
	}
	if digest := cluster.Bytes(); !p.consensus(digest) {
		log.Errorf("diverge proposal detected among %d peers! I proposed %s", len(cluster.Workers), cluster.Workers)
		return false, true
	}
	{
		stage := run.Stage{
			Version: p.clusterVersion + 1,
			Cluster: cluster,
		}
		// FIXME: assuming runners are up and running
		if err := execution.Par(cluster.Runners, func(ctrl plan.PeerID) error {
			return p.router.Send(ctrl.WithName("update"), stage.Encode(), connection.ConnControl, 0)
		}); err != nil {
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
	return true, keep
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
	changed, keep := p.propose(*cluster)
	if keep {
		p.Update()
	}
	return changed, keep, nil
}

func (p *Peer) getClusterConfig(url string) (*plan.Cluster, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("KungFu Peer: %s", p.self))
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(resp.Status)
	}
	defer resp.Body.Close()
	var cluster plan.Cluster
	if err = json.NewDecoder(resp.Body).Decode(&cluster); err != nil {
		return nil, err
	}
	return &cluster, nil
}
