package kungfu

import (
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

type Config struct {
	Algo         kb.KungFu_AllReduceAlgo
	ReportPeriod time.Duration
}

func (c Config) complete() Config {
	newConfig := Config{
		Algo:         c.Algo,
		ReportPeriod: c.ReportPeriod,
	}
	if newConfig.ReportPeriod == 0 {
		newConfig.ReportPeriod = 30 * time.Second
	}
	return newConfig
}

type Kungfu struct {
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
	session := newSession(config, ps, router)
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	localServer, err := rch.NewLocalServer(router)
	if err != nil {
		return nil, err
	}
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
	return 0
}

func (kf *Kungfu) Close() int {
	kf.server.Close() // TODO: check error
	kf.localServer.Close()
	return 0
}

func (kf *Kungfu) CurrentSession() *session {
	return kf.currentSession
}

func (kf *Kungfu) DebugCurrentCluster() *plan.ProcSpec {
	return kf.currentSession.cluster
}
