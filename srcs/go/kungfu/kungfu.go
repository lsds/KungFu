package kungfu

import (
	"errors"
	"fmt"
	"os"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/store"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Config struct {
	Algo kb.KungFu_AllReduceAlgo
}

func (c Config) complete() Config {
	newConfig := Config{
		Algo: c.Algo,
	}
	return newConfig
}

type Kungfu struct {
	sync.Mutex

	configClient   *ConfigClient
	self           *plan.PeerID
	currentSession *session

	store       *store.VersionedStore
	router      *rch.Router
	server      *rch.Server
	localServer *rch.Server
	config      Config
}

func New(config Config) (*Kungfu, error) {
	configClient, err := NewDefaultConfigClient()
	if err != nil {
		return nil, err
	}
	self, err := plan.GetSelfFromEnv()
	if err != nil {
		return nil, err
	}
	store := store.NewVersionedStore(3)
	router := rch.NewRouter(*self, store)
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	localServer, err := rch.NewLocalServer(router)
	if err != nil {
		return nil, err
	}
	return &Kungfu{
		configClient: configClient,
		self:         self,
		store:        store,
		router:       router,
		server:       server,
		localServer:  localServer,
		config:       config.complete(),
	}, nil
}

func (kf *Kungfu) Start() int {
	go kf.server.Serve()
	go kf.localServer.Serve()
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
	kf.localServer.Close()
	return 0
}

func (kf *Kungfu) StartStep(version string) int {
	var globalStep int
	if err := kf.configClient.GetConfig(version, kb.InitStepEnvKey, &globalStep); err != nil {
		err := fmt.Errorf("failed to get %s@%s: %v", kb.InitStepEnvKey, version, err)
		utils.ExitErr(err)
	}
	return globalStep
}

func (kf *Kungfu) ProposeUpdate(globalStep int, version string, newSize int) (bool, error) {
	log.Infof("Kungfu::ProposeUpdate with (%d@%q, %d)", globalStep, version, newSize)
	var hostSpecs []plan.HostSpec
	if err := kf.configClient.GetConfig("0", kb.HostSpecEnvKey, &hostSpecs); err != nil {
		log.Errorf("failed to get %s: %v", kb.HostSpecEnvKey, err)
		return false, err
	}
	log.Infof("generating new cluster spec of size %d", newSize)
	cs, err := plan.GenPeerList(newSize, hostSpecs)
	if err != nil {
		log.Errorf("failed to generate new cluster spec: %v", err)
		return false, err
	}
	if err := kf.configClient.PutConfig(version, kb.PeerListEnvKey, cs); err != nil {
		log.Warnf("failed to write config: %v", err)
		return false, err
	}
	if err := kf.configClient.PutConfig(version, kb.InitStepEnvKey, globalStep); err != nil {
		log.Warnf("failed to write config: %v", err)
		return false, err
	}
	_, keep := cs.Lookup(*kf.self)
	return keep, nil
}

var errSelfNotInCluster = errors.New("self not in cluster")

func (kf *Kungfu) CurrentSession() *session {
	kf.Lock()
	defer kf.Unlock()
	if kf.currentSession == nil {
		initSession := os.Getenv(kb.InitSessEnvKey)
		if exist := kf.updateSession(initSession); !exist {
			utils.ExitErr(errSelfNotInCluster)
		}
	}
	return kf.currentSession
}

func (kf *Kungfu) UpdateSession(version string) bool {
	kf.Lock()
	defer kf.Unlock()
	return kf.updateSession(version)
}

func (kf *Kungfu) updateSession(version string) bool {
	log.Infof("Kungfu::updateSession with version %q", version)
	var pl plan.PeerList
	if err := kf.configClient.GetConfig(version, kb.PeerListEnvKey, &pl); err != nil {
		log.Warnf("failed to get config: %v, running in single mode", err)
		pl = []plan.PeerID{*kf.self}
		// utils.ExitErr(err)
	}
	log.Infof("creating session of %d peers", len(pl))
	kf.router.ResetConnections()
	sess, exist, err := newSession(kf.config, kf.self, pl, kf.router)
	if !exist {
		return false
	}
	if err != nil {
		utils.ExitErr(err)
	}
	kf.currentSession = sess
	return true
}

func (kf *Kungfu) Save(version, name string, buf *kb.Buffer) int {
	blob := &store.Blob{Data: buf.Data}
	return code(kf.store.Create(version, name, blob))
}
