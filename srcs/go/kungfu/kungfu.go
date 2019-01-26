package kungfu

import (
	"fmt"
	"os"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/metrics"
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
	initProcSpec *plan.ProcSpec
	initSession  *session
	router       *rch.Router
	server       *rch.Server
	localServer  *rch.Server
	config       Config
}

func New(config Config) (*Kungfu, error) {
	config = config.complete()
	ps, err := plan.NewProcSpecFromEnv()
	if err != nil {
		return nil, err
	}
	session := newSession(config, ps)
	router := rch.NewRouter(ps.Self())
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	localServer, err := rch.NewLocalServer(router)
	if err != nil {
		return nil, err
	}
	return &Kungfu{
		initProcSpec: ps,
		initSession:  session,
		router:       router,
		server:       server,
		localServer:  localServer,
		config:       config,
	}, nil
}

func (kf *Kungfu) Start() int {
	go metrics.ListenAndServe(kf.currentCluster().MyMonitoringPort())
	go kf.server.Serve()
	go kf.localServer.Serve()
	go func() {
		for range time.Tick(kf.config.ReportPeriod) {
			kf.router.UpdateRate()
		}
	}()
	if kc.RunWarmup {
		return kf.Warmup()
	}
	return 0
}

func (kf *Kungfu) Close() int {
	kf.server.Close() // TODO: check error
	kf.localServer.Close()
	filename := fmt.Sprintf("vars.%02d.json", kf.currentCluster().MyRank())
	f, err := os.Create(filename)
	if err != nil {
		return 1
	}
	defer f.Close()
	metrics.RecordStop()
	metrics.Export(f)
	return 0
}

func (kf *Kungfu) currentCluster() *plan.ProcSpec {
	// TODO: get cluster by version
	return kf.initProcSpec
}
