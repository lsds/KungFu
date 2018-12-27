package kungfu

import (
	"fmt"
	"os"
	"time"

	"github.com/luomai/kungfu/srcs/go/metrics"
	rch "github.com/luomai/kungfu/srcs/go/rchannel"
	"github.com/luomai/kungfu/srcs/go/wire"
)

type Kungfu struct {
	cluster *rch.Cluster
	router  *rch.Router
	server  *rch.Server

	algo         wire.KungFu_AllReduceAlgo
	reportPeriod time.Duration
}

func New(algo wire.KungFu_AllReduceAlgo) (*Kungfu, error) {
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
	return &Kungfu{
		cluster:      cluster,
		router:       router,
		server:       server,
		algo:         algo,
		reportPeriod: 5 * time.Second,
	}, nil
}

func (kf *Kungfu) Start() int {
	go metrics.ListenAndServe(kf.cluster.MyMonitoringPort())
	go kf.server.ListenAndServe()
	go func() {
		for range time.Tick(kf.reportPeriod) {
			kf.router.UpdateRate()
		}
	}()
	return 0
}

func (kf *Kungfu) Close() int {
	kf.server.Close() // TODO: check error
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
