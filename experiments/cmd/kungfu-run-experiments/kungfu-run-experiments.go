package main

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/lsds/KungFu/experiments/tfkeras"
	"github.com/lsds/KungFu/srcs/go/kungfu/base"
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

	strategy base.Strategy
}{
	hostfile:   flag.String("hostfile", "hosts.txt", ""),
	logDir:     flag.String("logdir", ".", ""),
	usr:        flag.String("u", "", "user name for ssh"),
	verboseLog: flag.Bool("v", true, "show task log"),

	strategy: base.DefaultStrategy,
}

func init() {
	flag.Var(&flg.strategy, "strategy", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(base.StrategyNames(), " | ")))
}

func main() {
	flag.Parse()
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
	peers, err := c.Hostlist.GenPeerList(c.Size, pr)
	if err != nil {
		return err
	}

	ctx := context.TODO()
	j := e.Job(flg.strategy, c.Hostlist, pr, *flg.logDir)
	fmt.Printf("%s\n", j.DebugString())

	procs := j.CreateAllProcs(peers)
	d, err := utils.Measure(func() error {
		return remote.RemoteRunAll(ctx, *flg.usr, procs, *flg.verboseLog, *flg.logDir)
	})
	log.Infof("took %s", d)
	return err
}
