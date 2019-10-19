package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"runtime"
	"strings"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/remote"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of peers")
	hostList   = flag.String("H", plan.DefaultHostSpec.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	portRange  = flag.String("port-range", plan.DefaultPortRange.String(), "port range for the peers")
	user       = flag.String("u", "", "user name for ssh")
	timeout    = flag.Duration("timeout", 0, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	strategy   = flag.String("strategy", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.StrategyNames(), " | ")))
	checkpoint = flag.String("checkpoint", "0", "")
)

func init() {
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
}

func main() {
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	hl, err := plan.ParseHostList(*hostList)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	pr, err := plan.ParsePortRange(*portRange)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -port-range: %v", err))
	}
	jc := sch.JobConfig{
		Strategy:  kb.ParseStrategy(*strategy),
		HostList:  hl,
		PortRange: *pr,
		Prog:      restArgs[0],
		Args:      restArgs[1:],
	}
	ps, _, err := jc.CreateProcs(*np)
	if err != nil {
		utils.ExitErr(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	d, err := utils.Measure(func() error {
		_, err := runner.RemoteRunAll(ctx, *user, ps, *verboseLog)
		return err
	})
	log.Infof("all %d peers finished, took %s", len(ps), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
