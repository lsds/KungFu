package kungfu

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
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

	configClient   *configClient
	self           *plan.PeerSpec
	currentSession *session
	router         *rch.Router
	server         *rch.Server
	localServer    *rch.Server
	config         Config
}

func New(config Config) (*Kungfu, error) {
	configClient, err := newConfigClient()
	if err != nil {
		return nil, err
	}
	self, err := plan.GetSelfFromEnv()
	if err != nil {
		return nil, err
	}
	router := rch.NewRouter(*self)
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
		monitor.StartServer(int(kf.self.MonitoringPort))
		monitorAddr := plan.NetAddr{
			Host: kf.self.NetAddr.Host, // FIXME: use pubAddr
			Port: kf.self.MonitoringPort,
		}
		log.Infof("Kungfu peer %s started, monitoring endpoint http://%s/metrics", kf.self.NetAddr, monitorAddr)
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

func (kf *Kungfu) CurrentSession() *session {
	if kf.currentSession == nil {
		kf.updateSession()
	}
	return kf.currentSession
}

func (kf *Kungfu) updateSession() {
	kf.Lock()
	defer kf.Unlock()
	var cs plan.ClusterSpec
	if err := kf.configClient.getConfig(kb.ClusterSpecEnvKey, &cs); err != nil {
		log.Warnf("failed to get config: %v, running in single mode", err)
		cs = plan.ClusterSpec{Peers: []plan.PeerSpec{*kf.self}}
		// utils.ExitErr(err)
	}
	sess, err := newSession(kf.config, kf.self, &cs, kf.router)
	if err != nil {
		utils.ExitErr(err)
	}
	kf.currentSession = sess
}
