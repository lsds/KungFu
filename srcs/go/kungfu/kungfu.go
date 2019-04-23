package kungfu

import (
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
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

	self           plan.PeerSpec
	currentSession *session
	server         *rch.Server
	localServer    *rch.Server
	config         Config
}

func New(config Config) (*Kungfu, error) {
	config = config.complete()
	ps, err := plan.NewProcSpecFromEnv()
	if err != nil {
		return nil, err
	}
	self := ps.Self()
	router := rch.NewRouter(self)
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	localServer, err := rch.NewLocalServer(router)
	if err != nil {
		return nil, err
	}
	session := newSession(config, ps, router)
	return &Kungfu{
		self:           self,
		currentSession: session,
		server:         server,
		localServer:    localServer,
		config:         config,
	}, nil
}

func (kf *Kungfu) Start() int {
	go kf.server.Serve()
	go kf.localServer.Serve()
	if kc.RunWarmup {
		return kf.currentSession.Warmup()
	}
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
	kf.Lock()
	defer kf.Unlock()
	return kf.currentSession
}

func (kf *Kungfu) updateSession() {
	kf.Lock()
	defer kf.Unlock()
	// TODO:
	// kf.currentSession = newSession()
}
