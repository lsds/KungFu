package kungfu

import (
	"fmt"
	"os"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/metrics"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

type Kungfu struct {
	cluster     *rch.Cluster
	router      *rch.Router
	server      *rch.Server
	localServer *rch.Server
	config      Config
}

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

func New(config Config) (*Kungfu, error) {
	cluster, err := rch.NewClusterFromEnv()
	if err != nil {
		return nil, err
	}
	router, err := rch.NewRouter(cluster)
	if err != nil {
		return nil, err
	}
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	localServer, err := rch.NewLocalServer(router)
	if err != nil {
		return nil, err
	}
	return &Kungfu{
		cluster:     cluster,
		router:      router,
		server:      server,
		localServer: localServer,
		config:      config.complete(),
	}, nil
}

func (kf *Kungfu) Start() int {
	go metrics.ListenAndServe(kf.cluster.MyMonitoringPort())
	go kf.server.Serve()
	go kf.localServer.Serve()
	go func() {
		for range time.Tick(kf.config.ReportPeriod) {
			kf.router.UpdateRate()
		}
	}()
	kf.Warmup()
	return 0
}

func (kf *Kungfu) Close() int {
	kf.server.Close() // TODO: check error
	kf.localServer.Close()
	filename := fmt.Sprintf("vars.%02d.json", kf.cluster.MyRank())
	f, err := os.Create(filename)
	if err != nil {
		return 1
	}
	defer f.Close()
	metrics.RecordStop()
	metrics.Export(f)
	return 0
}
