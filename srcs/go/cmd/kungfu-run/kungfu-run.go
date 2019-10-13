package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"runtime"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of peers")
	hostList   = flag.String("H", plan.DefaultHostSpec.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	portRange  = flag.String("port-range", plan.DefaultPortRange.String(), "port range for the peers")
	selfHost   = flag.String("self", "", "internal IP")
	timeout    = flag.Duration("timeout", 0, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	nicName    = flag.String("nic", "", "network interface name, for infer self IP")
	algo       = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.StrategyNames(), " | ")))

	port        = flag.Int("port", 38080, "port for rchannel")
	watch       = flag.Bool("w", false, "watch config")
	watchPeriod = flag.Duration("watch-period", 500*time.Millisecond, "")
	checkpoint  = flag.String("checkpoint", "0", "")

	logfile = flag.String("logfile", "", "path to log file")
)

func init() {
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
	utils.LogNICInfo()
	utils.LogCudaEnv()
	utils.LogNCCLEnv()
}

var (
	errMissingProgramName = errors.New("missing program name")
)

func progName() string {
	if len(os.Args) > 0 {
		return path.Base(os.Args[0])
	}
	return ""
}

func main() {
	if len(*logfile) > 0 {
		lf, err := os.Create(*logfile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer lf.Close()
		log.SetOutput(lf)
	}
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s took %s", prog, time.Since(t0)) }(progName())
	selfIP, err := run.InferSelfIPv4(*selfHost, *nicName)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Infof("Using selfHost=%s", plan.FormatIPv4(selfIP))
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errMissingProgramName)
	}
	pr, err := plan.ParsePortRange(*portRange)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -port-range: %v", err))
	}
	hl, err := run.ResolveHostList(*hostList, *nicName)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	parent := plan.PeerID{IPv4: selfIP, Port: uint16(*port)}
	parents := func() plan.PeerList {
		var ps plan.PeerList
		for _, h := range hl {
			ps = append(ps, plan.PeerID{IPv4: h.IPv4, Port: uint16(*port)})
		}
		return ps
	}()
	if _, ok := parents.Lookup(parent); !ok {
		utils.ExitErr(fmt.Errorf("%s not in %s", parent, parents))
	}
	jc := sch.JobConfig{
		Parent:    parent,
		HostList:  hl,
		PortRange: *pr,
		Prog:      restArgs[0],
		Args:      restArgs[1:],
	}
	ctx, cancel := context.WithCancel(context.Background())
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}

	if *watch {
		peers, err := hl.GenPeerList(*np, *pr)
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
		}
		ch := make(chan run.Stage, 1)
		ch <- run.Stage{Cluster: peers, Checkpoint: *checkpoint}
		watchRun(ctx, parent, parents, ch, jc)
	} else {
		procs, _, err := jc.CreateProcs(*np, kb.ParseStrategy(*algo))
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to create tasks: %v", err))
		}
		simpleRun(ctx, selfIP, procs, jc)
	}
}

func simpleRun(ctx context.Context, selfIP uint32, ps []sch.Proc, jc sch.JobConfig) {
	myPs := sch.ForHost(selfIP, ps)
	if len(myPs) <= 0 {
		log.Infof("No task to run on this node")
		return
	}
	log.Infof("will parallel run %d instances of %s with %q", len(myPs), jc.Prog, jc.Args)
	d, err := utils.Measure(func() error { return runner.LocalRunAll(ctx, myPs, *verboseLog) })
	log.Infof("all %d/%d local peers finished, took %s", len(myPs), len(ps), d)
	if err != nil {
		utils.ExitErr(err)
	}
}
