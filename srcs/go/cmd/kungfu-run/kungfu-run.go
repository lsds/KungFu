package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"runtime"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of peers")
	hostList   = flag.String("H", plan.DefaultHostSpec().String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	selfHost   = flag.String("self", "", "internal IP")
	timeout    = flag.Duration("timeout", 0, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	nicName    = flag.String("nic", "", "network interface name, for infer self IP")
	algo       = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.AllAlgoNames(), " | ")))

	port        = flag.Int("port", 38080, "port for rchannel")
	watch       = flag.Bool("w", false, "watch config")
	watchPeriod = flag.Duration("watch-period", 500*time.Millisecond, "")
	keep        = flag.Bool("k", false, "don't stop watch")
	checkpoint  = flag.String("checkpoint", "0", "")
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

func main() {
	selfIP := func() string {
		switch {
		case len(*selfHost) > 0:
			return *selfHost
		case len(*nicName) > 0:
			return inferIP(*nicName)
		}
		return "127.0.0.1"
	}()
	log.Infof("Using selfHost=%s", selfIP)
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errMissingProgramName)
	}
	prog := restArgs[0]
	args := restArgs[1:]
	hl, err := plan.ParseHostList(*hostList)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	parent := plan.PeerID{Host: selfIP, Port: uint16(*port)}
	jc := sch.JobConfig{
		PeerCount: *np,
		Parent:    parent,
		HostList:  hl,
		Prog:      prog,
		Args:      args,
	}
	procs, peers, err := jc.CreateProcs(kb.ParseAlgo(*algo))
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to create tasks: %v", err))
	}

	ctx, cancel := context.WithCancel(context.Background())
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}

	if *watch {
		ch := make(chan run.Stage, 1)
		ch <- run.Stage{Cluster: peers, Checkpoint: *checkpoint}
		server, err := rch.NewServer(run.NewHandler(parent, ch))
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to create server: %v", err))
		}
		go server.Serve()
		defer server.Close()
		watchRun(ctx, selfIP, ch, jc)
	} else {
		simpleRun(ctx, selfIP, procs, prog, args)
	}
}

func simpleRun(ctx context.Context, selfIP string, ps []sch.Proc, prog string, args []string) {
	myPs := sch.ForHost(selfIP, ps)
	if len(myPs) <= 0 {
		log.Infof("No task to run on this node")
		return
	}
	log.Infof("will parallel run %d instances of %s with %q", len(myPs), prog, args)
	d, err := utils.Measure(func() error { return runner.LocalRunAll(ctx, myPs, *verboseLog) })
	log.Infof("all %d/%d local peers finished, took %s", len(myPs), len(ps), d)
	if err != nil && err != context.DeadlineExceeded {
		utils.ExitErr(err)
	}
}

func inferIP(nicName string) string {
	ifaces, err := net.Interfaces()
	if err != nil {
		return "127.0.0.1"
	}
	for _, i := range ifaces {
		if i.Name != nicName {
			continue
		}
		addrs, err := i.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			var ip net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip.To4() != nil {
				return ip.String()
			}
		}
	}
	return "127.0.0.1"
}
