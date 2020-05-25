package main

import (
	"context"
	"flag"
	"fmt"
	"strconv"
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
	hostfile     *string
	clusterSizes *string

	quiet      *bool
	logDir     *string
	usr        *string
	verboseLog *bool
	nic        *string
	kfRoot     *string

	strategy base.Strategy
}{
	hostfile:     flag.String("hostfile", "hosts.txt", ""),
	clusterSizes: flag.String("cluster-sizes", "", ""),

	quiet:      flag.Bool("q", false, ""),
	logDir:     flag.String("logdir", ".", ""),
	usr:        flag.String("u", "", "user name for ssh"),
	nic:        flag.String("nic", "", ""),
	verboseLog: flag.Bool("v", true, "show task log"),
	kfRoot:     flag.String("kf-root", "./.kungfu/KungFu", ""),

	strategy: base.DefaultStrategy,
}

func init() {
	flag.Var(&flg.strategy, "strategy", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(base.StrategyNames(), " | ")))
}

func main() {
	flag.Parse()
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s finished, took %s", prog, time.Since(t0)) }(utils.ProgName())
	sizes, err := parseIntList(*flg.clusterSizes)
	if err != nil {
		utils.ExitErr(err)
	}
	hl, err := hostfile.ParseFile(*flg.hostfile)
	if err != nil {
		utils.ExitErr(err)
	}
	fmt.Printf("total cap: %d\n", hl.Cap())
	for i, h := range hl {
		fmt.Printf("host[%d]=%s\n", i, h.DebugString())
	}

	cs := generateClusters(hl, sizes)
	es := tfkeras.Default()
	succ, failed := combine(cs, es, run)
	fmt.Printf("run %d experiments, succ: %d, failed: %d\n", succ+failed, succ, failed)
}

func combine(cs []Cluster, es []tfkeras.Experiment, f func(Cluster, tfkeras.Experiment) error) (int, int) {
	var idx int
	var succ, failed int
	for _, c := range cs {
		log.Infof("will runn %d experiments with %d peers", len(es), c.Size)
		for _, e := range es {
			idx++
			if err := f(c, e); err != nil {
				log.Errorf("experiment #%d failed: %v", idx, err)
				failed++
			} else {
				succ++
			}
		}
	}
	return succ, failed
}

func run(c Cluster, e tfkeras.Experiment) error {
	pr := plan.DefaultPortRange
	ctx := context.TODO()
	j := e.Job(*flg.kfRoot, flg.strategy, c.Hostlist, pr, *flg.logDir)
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
		return remote.RunStaticKungFuJob(ctx, j, sp, *flg.quiet)
	})
	log.Infof("run tfkeras.Experiment took %s", d)
	return err
}

func parseIntList(line string) ([]int, error) {
	var ns []int
	for _, s := range strings.Split(line, ",") {
		n, err := strconv.Atoi(s)
		if err != nil {
			return nil, err
		}
		ns = append(ns, n)
	}
	return ns, nil
}
