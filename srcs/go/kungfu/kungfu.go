package kungfu

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Kungfu struct {
	sync.Mutex

	// immutable
	configServerURL    string
	initClusterVersion int
	parent             plan.PeerID
	self               plan.PeerID
	strategy           kb.Strategy
	single             bool
	router             *rch.Router
	server             rch.Server
	client             http.Client

	// dynamic
	clusterVersion int
	currentSession *session
	currentCluster *plan.Cluster
	updated        bool
}

func New() (*Kungfu, error) {
	config, err := plan.ParseConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return NewFromConfig(config)
}

func NewFromConfig(config *plan.Config) (*Kungfu, error) {
	router := rch.NewRouter(config.Self)
	server := rch.NewServer(router)
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
	return &Kungfu{
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

func (kf *Kungfu) Start() error {
	if !kf.single {
		if err := kf.server.Start(); err != nil {
			return err
		}
		if kc.EnableMonitoring {
			monitoringPort := kf.self.Port + 10000
			monitor.StartServer(int(monitoringPort))
			monitorAddr := plan.NetAddr{
				IPv4: kf.self.IPv4, // FIXME: use pubAddr
				Port: monitoringPort,
			}
			log.Infof("Kungfu peer %s started, monitoring endpoint http://%s/metrics", kf.self, monitorAddr)
		}
	}
	kf.Update()
	return nil
}

func (kf *Kungfu) Close() error {
	if !kf.single {
		if kc.EnableMonitoring {
			monitor.StopServer()
		}
		kf.server.Close() // TODO: check error
	}
	return nil
}

// UID returns an immutable unique ID of this peer
func (kf *Kungfu) UID() uint64 {
	hi := uint64(kf.self.IPv4)
	lo := (uint64(kf.self.Port) << 16) | uint64(uint16(kf.initClusterVersion))
	return (hi<<32 | lo)
}

var errSelfNotInCluster = errors.New("self not in cluster")

func (kf *Kungfu) CurrentSession() *session {
	kf.Lock()
	defer kf.Unlock()
	if kf.currentSession == nil {
		kf.updateTo(kf.currentCluster.Workers)
	}
	return kf.currentSession
}

func (kf *Kungfu) Update() bool {
	kf.Lock()
	defer kf.Unlock()
	return kf.updateTo(kf.currentCluster.Workers)
}

func (kf *Kungfu) updateTo(pl plan.PeerList) bool {
	if kc.EnableStallDetection {
		name := fmt.Sprintf("updateTo(%s)", pl.DebugString())
		defer utils.InstallStallDetector(name).Stop()
	}
	kf.server.SetToken(uint32(kf.clusterVersion))
	if kf.updated {
		log.Debugf("ignore update")
		return true
	}
	log.Debugf("Kungfu::updateTo v%d of %d peers: %s", kf.clusterVersion, len(pl), pl)
	kf.router.ResetConnections(pl, uint32(kf.clusterVersion))
	sess, exist := newSession(kf.strategy, kf.self, pl, kf.router)
	if !exist {
		return false
	}
	if err := sess.barrier(); err != nil {
		utils.ExitErr(fmt.Errorf("barrier failed after newSession: %v", err))
	}
	kf.currentSession = sess
	kf.updated = true
	return true
}

func (kf *Kungfu) SaveVersion(version, name string, buf *kb.Vector) error {
	return kf.router.P2P.SaveVersion(version, name, buf)
}

func (kf *Kungfu) Save(name string, buf *kb.Vector) error {
	return kf.router.P2P.Save(name, buf)
}

func par(ps plan.PeerList, f func(plan.PeerID) error) error {
	errs := make([]error, len(ps))
	var wg sync.WaitGroup
	for i, p := range ps {
		wg.Add(1)
		go func(i int, p plan.PeerID) {
			errs[i] = f(p)
			wg.Done()
		}(i, p)
	}
	wg.Wait()
	return mergeErrors(errs, "par")
}

func (kf *Kungfu) consensus(bs []byte) bool {
	sess := kf.CurrentSession()
	ok, err := sess.BytesConsensus(bs, "")
	if err != nil {
		utils.ExitErr(err)
	}
	return ok
}

func (kf *Kungfu) propose(cluster plan.Cluster) (bool, bool) {
	if kf.currentCluster.Eq(cluster) {
		log.Debugf("ingore unchanged proposal")
		return false, true
	}
	if digest := cluster.Bytes(); !kf.consensus(digest) {
		log.Errorf("diverge proposal detected among %d peers! I proposed %s", len(cluster.Workers), cluster.Workers)
		return false, true
	}
	{
		stage := run.Stage{
			Version: kf.clusterVersion + 1,
			Cluster: cluster,
		}
		// FIXME: assuming runners are up and running
		if err := par(cluster.Runners, func(ctrl plan.PeerID) error {
			return kf.router.Send(ctrl.WithName("update"), stage.Encode(), connection.ConnControl, 0)
		}); err != nil {
			utils.ExitErr(err)
		}
	}
	func() {
		kf.Lock()
		defer kf.Unlock()
		if kf.currentCluster.Workers.Disjoint(cluster.Workers) {
			log.Errorf("Full update detected: %s -> %s! State will be lost.", kf.currentCluster.DebugString(), cluster.DebugString())
		} else if len(cluster.Workers) > 0 && !kf.currentCluster.Workers.Contains(cluster.Workers[0]) {
			log.Errorf("New root can't not be a new worker! State will be lost.")
		}
		kf.currentCluster = &cluster
		kf.clusterVersion++
		kf.updated = false
	}()
	_, keep := cluster.Workers.Rank(kf.self)
	return true, keep
}

func (kf *Kungfu) ResizeClusterFromURL() (bool, bool, error) {
	var cluster *plan.Cluster
	for i := 0; ; i++ {
		var err error
		cluster, err = kf.getClusterConfig(kf.configServerURL)
		if err != nil {
			log.Errorf("getClusterConfig failed: %v, using current config", err)
			cluster = kf.currentCluster
		}
		if digest := cluster.Bytes(); kf.consensus(digest) {
			if i > 0 {
				log.Infof("New peer list is consistent after failed %d times", i)
			} else {
				log.Debugf("New peer list is consistent after ONE attempt!")
			}
			break
		}
		log.Warnf("diverge proposal detected among %d peers! I proposed %s", len(kf.currentCluster.Workers), cluster.DebugString())
		time.Sleep(50 * time.Millisecond)
	}
	changed, keep := kf.propose(*cluster)
	if keep {
		kf.Update()
	}
	return changed, keep, nil
}

func (kf *Kungfu) getClusterConfig(url string) (*plan.Cluster, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("KungFu Peer: %s", kf.self))
	resp, err := kf.client.Do(req)
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
