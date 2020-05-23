package main

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/lsds/KungFu/experiments/tfkeras"
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/kungfu/runtime"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/hostfile"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
)

var flg = struct {
	hostfile   *string
	logDir     *string
	usr        *string
	verboseLog *bool
	nic        *string

	strategy base.Strategy
}{
	hostfile:   flag.String("hostfile", "hosts.txt", ""),
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
	hl, err := hostfile.ParseFile(*flg.hostfile)
	if err != nil {
		utils.ExitErr(err)
	}
	fmt.Printf("total cap: %d\n", hl.Cap())
	for _, h := range hl {
		fmt.Printf("%s\n", h)
	}

	// sizes := []int{1, 2, 4, 8, 8 * 2, 8 * 3, 8 * 4, 8 * 5, 8 * 6}
	sizes := []int{4}
	cs := generateClusters(hl, sizes)
	es := tfkeras.Default()
	tailed := combine(cs, es, run)
	fmt.Printf("failed %d experiments\n", tailed)
}

func combine(cs []Cluster, es []tfkeras.Experiment, f func(Cluster, tfkeras.Experiment) error) int {
	var idx int
	var failed int
	for _, c := range cs {
		log.Infof("will runn %d experiments with %d peers", len(es), c.Size)
		for _, e := range es {
			idx++
			if err := f(c, e); err != nil {
				log.Errorf("experiment #%d failed: %v", idx, err)
				failed++
			}
		}
	}
	return failed
}

func run(c Cluster, e tfkeras.Experiment) error {
	pr := plan.DefaultPortRange
	ctx := context.TODO()
	j := e.Job(flg.strategy, c.Hostlist, pr, *flg.logDir)
	fmt.Printf("%s\n", j.DebugString())
	sp := runtime.SystemParameters{
		User:            *flg.usr,
		WorkerPortRange: pr,
		RunnerPort:      plan.DefaultRunnerPort,
		HostList:        c.Hostlist,
		ClusterSize:     c.Size,
		Nic:             *flg.nic,
	}
	d, err := utils.Measure(func() error {
		return remote.RunStaticKungFuJob(ctx, j, sp)
	})
	log.Infof("took %s", d)
	return err
}
