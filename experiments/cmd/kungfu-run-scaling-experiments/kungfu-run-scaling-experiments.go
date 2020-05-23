package main

import (
	"context"
	"flag"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/lsds/KungFu/experiments/elastic"
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/runtime"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/hostfile"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
)

type Cluster struct {
	Hostlist plan.HostList
	Size     int
}

var flg = struct {
	hostfile *string
	id       *string

	logDir     *string
	usr        *string
	verboseLog *bool
	nic        *string

	strategy base.Strategy
}{
	hostfile: flag.String("hostfile", "hosts.txt", ""),
	id:       flag.String("job-id", strconv.Itoa(int(time.Now().Unix())), ""),

	logDir:     flag.String("logdir", ".", ""),
	usr:        flag.String("u", "", "user name for ssh"),
	nic:        flag.String("nic", "", ""),
	verboseLog: flag.Bool("v", true, "show task log"),

	strategy: base.DefaultStrategy,
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
		Size:     hl.Cap(),
	}
	run(c)
}

func run(c Cluster) error {
	pr := plan.DefaultPortRange
	ctx := context.TODO()
	j := elastic.TestJob(*flg.id, flg.strategy, c.Hostlist, pr, *flg.logDir)
	log.Infof("will run %s\n", j.DebugString())
	sp := runtime.SystemParameters{
		User:            *flg.usr,
		WorkerPortRange: pr,
		RunnerPort:      plan.DefaultRunnerPort,
		HostList:        c.Hostlist,
		ClusterSize:     c.Size,
		Nic:             *flg.nic,
	}
	d, err := utils.Measure(func() error {
		return remote.RunElasticKungFuJob(ctx, j, sp)
	})
	log.Infof("took %s", d)
	return err
}
