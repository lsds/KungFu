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
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Kungfu struct {
	sync.Mutex

	// immutable
	configServerURL    string
	initClusterVersion int
	parent             plan.PeerID
	parents            plan.PeerList
	hostList           plan.HostList // FIXME: make it dynamic
	portRange          plan.PortRange
	self               plan.PeerID
	strategy           kb.Strategy
	single             bool
	router             *rch.Router
	server             rch.Server
	client             http.Client

	// dynamic
	clusterVersion int
	currentSession *session
	currentPeers   plan.PeerList
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
	return &Kungfu{
		configServerURL:    config.ConfigServer,
		parent:             config.Parent,
		parents:            config.Parents,
		currentPeers:       config.InitPeers,
		self:               config.Self,
		hostList:           config.HostList,
		portRange:          config.PortRange,
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
		kf.updateTo(kf.currentPeers)
	}
	return kf.currentSession
}

func (kf *Kungfu) Update() bool {
	kf.Lock()
	defer kf.Unlock()
	return kf.updateTo(kf.currentPeers)
}

func (kf *Kungfu) updateTo(pl plan.PeerList) bool {
	if kf.updated {
		log.Debugf("ignore update")
		return true
	}
	log.Debugf("Kungfu::updateTo v%d of %d peers: %s", kf.clusterVersion, len(pl), pl)
	kf.router.ResetConnections(pl)
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

func (kf *Kungfu) propose(peers plan.PeerList) (bool, bool) {
	if peers.Eq(kf.currentPeers) {
		log.Debugf("ingore unchanged proposal")
		return false, true
	}
	if digest := peers.Bytes(); !kf.consensus(digest) {
		log.Errorf("diverge proposal detected among %d peers! I proposed %s", len(kf.currentPeers), peers)
		return false, true
	}
	{
		stage := run.Stage{
			Version: kf.clusterVersion + 1,
			Cluster: peers,
		}
		if err := par(kf.parents, func(parent plan.PeerID) error {
			return kf.router.Send(parent.WithName("update"), stage.Encode(), rch.ConnControl, 0)
		}); err != nil {
			utils.ExitErr(err)
		}
	}
	func() {
		kf.Lock()
		defer kf.Unlock()
		kf.currentPeers = peers
		kf.clusterVersion++
		kf.updated = false
	}()
	_, keep := peers.Rank(kf.self)
	return true, keep
}

func (kf *Kungfu) ResizeCluster(newSize int) (bool, bool, error) {
	log.Debugf("resize cluster to %d", newSize)
	peers, err := kf.hostList.GenPeerList(newSize, kf.portRange)
	if err != nil {
		return false, true, err
	}
	changed, keep := kf.propose(peers)
	if keep {
		kf.Update()
	}
	return changed, keep, nil
}

func (kf *Kungfu) ResizeClusterFromURL() (bool, bool, error) {
	defer utils.InstallStallDetector("ResizeClusterFromURL").Stop()
	var peers plan.PeerList
	for i := 0; ; i++ {
		var err error
		peers, err = kf.getPeerListFromURL(kf.configServerURL)
		if err != nil {
			log.Errorf("getPeerListFromURL failed: %v, using current config", err)
			peers = kf.currentPeers
		}
		if digest := peers.Bytes(); kf.consensus(digest) {
			if i > 0 {
				log.Infof("New peer list is consistent after failed %d times", i)
			} else {
				log.Debugf("New peer list is consistent after ONE attempt!")
			}
			break
		}
		log.Warnf("diverge proposal detected among %d peers! I proposed %s", len(kf.currentPeers), peers)
		time.Sleep(50 * time.Millisecond)
	}
	changed, keep := kf.propose(peers)
	if keep {
		kf.Update()
	}
	return changed, keep, nil
}

func (kf *Kungfu) getPeerListFromURL(url string) (plan.PeerList, error) {
	resp, err := kf.client.Get(url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(resp.Status)
	}
	var o struct {
		Peers plan.PeerList `json:"peers"`
	}
	if err = json.NewDecoder(resp.Body).Decode(&o); err != nil {
		return nil, err
	}
	if kf.hostList.Cap() < len(o.Peers) {
		return nil, plan.ErrNoEnoughCapacity
	}
	return o.Peers, nil
}
