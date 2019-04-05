package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/platforms/huawei"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	verboseLog = flag.Bool("v", true, "show task log")
	timeout    = flag.Duration("timeout", 10*time.Second, "timeout")
	algo       = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.AllAlgoNames(), " | ")))
)

func init() {
	utils.LogAllEnvs()

	log.SetPrefix("[kungfu-huawei-launcher] ")
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
}

func main() {
	flag.Parse()
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	prog := restArgs[0]
	args := restArgs[1:]

	env, err := huawei.ParseEnv()
	if err != nil {
		utils.ExitErr(err)
	}
	hosts := createHostSpec(env)
	log.Printf("using hostspec: %s", plan.FormatHostSpec(hosts))
	jc := sch.JobConfig{
		PeerCount: plan.TotalCap(hosts),
		HostList:  plan.FormatHostSpec(hosts),
		Prog:      prog,
		Args:      args,
	}
	ps, err := jc.CreateProcs(kb.ParseAlgo(*algo))
	if err != nil {
		utils.ExitErr(err)
	}
	selfIP := env.Peers[env.ContainerIndex]
	myPs := sch.ForHost(selfIP, ps)
	if len(myPs) <= 0 {
		log.Print("No task to run on this node")
		return
	}
	log.Printf("will parallel run %d instances of %s with %q", len(myPs), prog, args)

	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	d, err := utils.Measure(func() error { return runner.LocalRunAll(ctx, myPs, *verboseLog) })
	log.Printf("all %d/%d local peers finished, took %s", len(myPs), len(ps), d)
	if err != nil && err != context.DeadlineExceeded {
		utils.ExitErr(err)
	}
}

func createHostSpec(env *huawei.ContainerInfo) []plan.HostSpec {
	var hs []plan.HostSpec
	for _, h := range env.Peers {
		hs = append(hs, plan.HostSpec{
			Hostname:   h,
			Slots:      1, // FIXME: support multi-gpu in one container
			PublicAddr: h,
		})
	}
	return hs
}
