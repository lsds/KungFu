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
	"github.com/lsds/KungFu/srcs/go/platforms/huawei"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np          = flag.Int("np", 8, "number of peers")
	verboseLog  = flag.Bool("v", true, "show task log")
	timeout     = flag.Duration("timeout", 10*time.Second, "timeout")
	algo        = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.AllAlgoNames(), " | ")))
	disableNCCL = flag.Bool("disable-nccl", true, "disable NCCL")
)

func init() {
	// utils.LogAllEnvs()
	utils.LogCudaEnv()
	utils.LogKungfuEnv()
	utils.LogNICInfo()
	log.SetPrefix("[kungfu-huawei-launcher] ")
	flag.Parse()
	utils.LogArgs()
}

func main() {
	flag.Parse()
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	prog := restArgs[0]
	args := restArgs[1:]
	env, err := huawei.ParseEnv(*np)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Printf("GPUs: %s", strings.Join(env.GPUs, ","))
	// if err := utils.TestConnectivity(env.ClusterSpec, env.ContainerIndex); err != nil {
	// 	utils.ExitErr(err)
	// }
	ps, err := sch.CreateProcs(prog, args, env.ClusterSpec, kb.ParseAlgo(*algo), *disableNCCL)
	if err != nil {
		utils.ExitErr(err)
	}

	myPs := sch.ForHost(env.SelfIPv4, ps)
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
