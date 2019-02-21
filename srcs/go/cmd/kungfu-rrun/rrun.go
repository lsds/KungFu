package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"runtime"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/runner"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of peers")
	hostList   = flag.String("H", plan.DefaultHostSpec().String(), "comma separated list of <hostname>:<nslots>[:<public addr>]")
	user       = flag.String("u", "", "user name for ssh")
	timeout    = flag.Duration("timeout", 10*time.Second, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	algo       = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.AllAlgoNames(), " | ")))
)

func init() {
	log.SetPrefix("[kungfu-rrun] ")
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
}

func main() {
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	jc := sch.JobConfig{
		PeerCount: *np,
		HostList:  *hostList,
		Prog:      restArgs[0],
		Args:      restArgs[1:],
	}
	ps, err := jc.CreateProcs(kb.ParseAlgo(*algo))
	if err != nil {
		utils.ExitErr(err)
	}

	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	d, err := utils.Measure(func() error {
		_, err := runner.RemoteRunAll(ctx, *user, ps, *verboseLog)
		return err
	})
	log.Printf("all %d peers finished, took %s", len(ps), d)
	if err != nil && err != context.DeadlineExceeded {
		utils.ExitErr(err)
	}
}
