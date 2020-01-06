package kungfu

import (
	"errors"
	"fmt"
	"sync"

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
	parent    plan.PeerID
	parents   plan.PeerList
	hostList  plan.HostList
	portRange plan.PortRange
	self      plan.PeerID
	strategy  kb.Strategy
	single    bool

	router *rch.Router
	server rch.Server

	// dynamic
	currentSession *session
	currentPeers   plan.PeerList
	initStep       string
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
	return &Kungfu{
		parent:       config.Parent,
		parents:      config.Parents,
		currentPeers: config.InitPeers,
		self:         config.Self,
		hostList:     config.HostList,
		portRange:    config.PortRange,
		strategy:     config.Strategy,
		initStep:     config.InitStep,
		single:       config.Single,
		router:       router,
		server:       server,
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

var errSelfNotInCluster = errors.New("self not in cluster")

func (kf *Kungfu) CurrentSession() *session {
	kf.Lock()
	defer kf.Unlock()
	if kf.currentSession == nil {
		kf.updateTo(kf.currentPeers)
	}
	return kf.currentSession
}

func (kf *Kungfu) GetInitStep() string {
	return kf.initStep
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
	log.Debugf("Kungfu::updateTo(%s), %d peers", pl, len(pl))
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

func (kf *Kungfu) propose(initStep string, peers plan.PeerList) (bool, bool) {
	if peers.Eq(kf.currentPeers) {
		log.Debugf("ingore unchanged proposal")
		return false, true
	}
	if digest := peers.Bytes(); !kf.consensus(digest) {
		log.Errorf("diverge proposal detected! I proposed %s", peers)
		return false, true
	}
	{
		stage := run.Stage{InitStep: initStep, Cluster: peers}
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
		kf.initStep = initStep
		kf.updated = false
	}()
	_, keep := peers.Rank(kf.self)
	return true, keep
}

func (kf *Kungfu) ResizeCluster(initStep string, newSize int) (bool, bool, error) {
	log.Debugf("resize cluster to %d with checkpoint %q", newSize, initStep)
	peers, err := kf.hostList.GenPeerList(newSize, kf.portRange)
	if err != nil {
		return false, true, err
	}
	changed, keep := kf.propose(initStep, peers)
	if keep {
		kf.Update()
	}
	return changed, keep, nil
}
