package kungfu

import (
	"errors"
	"os"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/store"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Config struct {
	Strategy kb.Strategy
}

type Kungfu struct {
	sync.Mutex

	// immutable
	parent    plan.PeerID
	hostList  plan.HostList
	portRange plan.PortRange
	self      plan.PeerID

	store  *store.VersionedStore
	router *rch.Router
	server rch.Server
	config Config

	// dynamic
	currentSession *session
	currentPeers   plan.PeerList
	checkpoint     string
	updated        bool
}

func New(config Config) (*Kungfu, error) {
	env, err := plan.ParseEnv()
	if err != nil {
		return nil, err
	}
	store := store.NewVersionedStore(3)
	router := rch.NewRouter(env.Self, store)
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	return &Kungfu{
		parent:       env.Parent,
		currentPeers: env.InitPeers,
		self:         env.Self,
		hostList:     env.HostList,
		portRange:    env.PortRange,
		checkpoint:   os.Getenv(kb.CheckpointEnvKey),
		store:        store,
		router:       router,
		server:       server,
		config:       config,
	}, nil
}

func (kf *Kungfu) Start() int {
	go kf.server.Serve()
	if kc.EnableMonitoring {
		monitoringPort := kf.self.Port + 10000
		monitor.StartServer(int(monitoringPort))
		monitorAddr := plan.NetAddr{
			Host: kf.self.Host, // FIXME: use pubAddr
			Port: monitoringPort,
		}
		log.Infof("Kungfu peer %s started, monitoring endpoint http://%s/metrics", kf.self, monitorAddr)
	}
	return 0
}

func (kf *Kungfu) Close() int {
	if kc.EnableMonitoring {
		monitor.StopServer()
	}
	kf.server.Close() // TODO: check error
	return 0
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
	log.Debugf("Kungfu::updateTo(%s)", pl)
	kf.router.ResetConnections() // FIXME: don't reset all connections
	sess, exist, err := newSession(kf.config, kf.self, pl, kf.router)
	if !exist {
		return false
	}
	if err != nil {
		utils.ExitErr(err)
	}
	kf.currentSession = sess
	kf.updated = true
	return true
}

func (kf *Kungfu) Save(version, name string, buf *kb.Vector) int {
	blob := &store.Blob{Data: buf.Data}
	return code(kf.store.Create(version, name, blob))
}

func (kf *Kungfu) propose(ckpt string, peers plan.PeerList) bool {
	if peers.Eq(kf.currentPeers) {
		log.Debugf("ingore unchanged proposal")
		return true
	}
	{
		stage := run.Stage{Checkpoint: ckpt, Cluster: peers}
		// FIXME: use par
		for _, h := range kf.hostList {
			id := plan.PeerID{Host: h.Hostname, Port: kf.parent.Port}
			if err := kf.router.Send(id.WithName("update"), stage.Encode(), rch.ConnControl, 0); err != nil {
				utils.ExitErr(err)
			}
		}
	}
	func() {
		kf.Lock()
		defer kf.Unlock()
		kf.currentPeers = peers
		kf.checkpoint = ckpt
		kf.updated = false
	}()
	_, keep := peers.Lookup(kf.self)
	return keep
}

func (kf *Kungfu) ResizeCluster(ckpt string, newSize int) (bool, error) {
	log.Debugf("resize cluster to %d at %q", newSize, ckpt)
	peers, err := kf.hostList.GenPeerList(newSize, kf.portRange)
	if err != nil {
		return true, err
	}
	if keep := kf.propose(ckpt, peers); !keep {
		return false, nil
	}
	return kf.Update(), nil
}
