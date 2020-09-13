package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/lsds/KungFu/experiments/elastic"
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/runtime"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/hostfile"
	"github.com/lsds/KungFu/srcs/go/proc"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
	"github.com/lsds/KungFu/tests/go/configserver"
)

type Cluster struct {
	Hostlist plan.HostList
	Size     int
}

var flg = struct {
	hostfile *string
	id       *string

	quiet      *bool
	logDir     *string
	usr        *string
	verboseLog *bool
	nic        *string
	kfRoot     *string

	strategy base.Strategy

	batchSize *int
	epochs    *int
	epochSize *int

	schedule *string
}{
	hostfile: flag.String("hostfile", "hosts.txt", ""),
	id:       flag.String("job-id", strconv.Itoa(int(time.Now().Unix())), ""),

	quiet:      flag.Bool("q", false, ""),
	logDir:     flag.String("logdir", ".", ""),
	usr:        flag.String("u", "", "user name for ssh"),
	nic:        flag.String("nic", "", ""),
	verboseLog: flag.Bool("v", true, "show task log"),

	kfRoot:   flag.String("kf-root", "./.kungfu/KungFu", ""),
	strategy: base.DefaultStrategy,

	batchSize: flag.Int("batch-size", 1, ""),
	epochs:    flag.Int("epochs", 1, ""),
	epochSize: flag.Int("epoch-size", 100, ""),

	schedule: flag.String("resize-schedule", "", ""),
}

func init() {
	flag.Var(&flg.strategy, "strategy", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(base.StrategyNames(), " | ")))
}

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s finished, took %s", prog, time.Since(t0)) }(utils.ProgName())
	log.Infof("%s started", utils.ProgName())

	hl, err := hostfile.ParseFile(*flg.hostfile)
	if err != nil {
		utils.ExitErr(err)
	}
	fmt.Printf("total cap: %d\n", hl.Cap())
	for i, h := range hl {
		fmt.Printf("host[%d]=%s\n", i, h.DebugString())
	}

	c := Cluster{
		Hostlist: hl,
		Size:     1,
	}

	cfgPort := 9000
	cfgHost := plan.FormatIPv4(hl[0].IPv4)
	cancel, wg := runConfigServer(cfgHost, cfgPort)
	defer cancel()
	cfgServer := url.URL{
		Scheme: `http`,
		Host:   net.JoinHostPort(cfgHost, strconv.Itoa(cfgPort)),
		Path:   `/get`,
	}

	pr := plan.DefaultPortRange
	sp := runtime.SystemParameters{
		User:            *flg.usr,
		WorkerPortRange: pr,
		RunnerPort:      plan.DefaultRunnerPort,
		HostList:        c.Hostlist,
		ClusterSize:     c.Size,
		Nic:             *flg.nic,
	}

	cc := configserver.NewClient(cfgServer.String())
	{
		cc.WaitServer()
		cc.Reset()
		cluster := plan.Cluster{
			Runners: hl.GenRunnerList(sp.RunnerPort),
			Workers: hl.MustGenPeerList(hl.Cap(), sp.WorkerPortRange),
		}
		if err := cc.Update(cluster); err != nil {
			log.Errorf("%v", err)
		}
	}
	skip := false
	if !skip {
		e := elastic.Default()
		e.Epochs = *flg.epochs
		e.EpochSize = *flg.epochSize
		e.BatchSize = *flg.batchSize
		e.Schedule = *flg.schedule

		run(e, c, sp, cfgServer.String())
	} else {
		time.Sleep(3 * time.Second)
	}
	if err := cc.StopServer(); err != nil {
		log.Errorf("stopConfigServer: %v", err)
	}
	wg.Wait()
}

func run(e elastic.Experiment, c Cluster, sp runtime.SystemParameters, cfgServer string) error {
	ctx := context.TODO()
	j := e.Job(*flg.id, *flg.kfRoot, cfgServer, flg.strategy, c.Hostlist, sp.WorkerPortRange, *flg.logDir)
	log.Infof("will run %s", j.DebugString())

	d, err := utils.Measure(func() error {
		return remote.RunElasticKungFuJob(ctx, j, sp, *flg.quiet)
	})
	log.Infof("run elastic.Experiment took %s", d)
	return err
}

func runConfigServer(hostname string, port int) (context.CancelFunc, *sync.WaitGroup) {
	envs := make(proc.Envs)
	envs[`PATH`] = `$HOME/go/bin:$PATH`
	args := []string{`-port`, strconv.Itoa(port)}
	p := proc.Proc{
		Name:     `config-server`,
		Prog:     `kungfu-config-server`,
		Args:     args,
		Hostname: hostname,
		Envs:     envs,
	}
	ctx, cancel := context.WithCancel(context.TODO())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		if err := remote.RemoteRunAll(ctx, *flg.usr, []proc.Proc{p}, true, *flg.logDir); err != nil {
			log.Errorf("%s failed: %v", p.Name, err)
		}
		wg.Done()
	}()
	return cancel, &wg
}
