package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/remote"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	hostList   = flag.String("H", plan.DefaultHostSpec.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	timeout    = flag.Duration("timeout", 0, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	user       = flag.String("u", "", "user name for ssh")
)

func init() {
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
	utils.LogNICInfo()
	utils.LogCudaEnv()
	utils.LogNCCLEnv()
}

func progName() string {
	if len(os.Args) > 0 {
		return path.Base(os.Args[0])
	}
	return ""
}

func main() {
	t0 := time.Now()
	defer func(prog string) { log.Infof("%s took %s", prog, time.Since(t0)) }(progName())
	hl, err := plan.ParseHostList(*hostList)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	log.Infof("Using %d hosts, total slots: %d", len(hl), hl.Cap())
	ctx, cancel := context.WithCancel(context.Background())
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	args := flag.Args()
	if len(args) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	distribute(ctx, hl, args[0], args[1:])
}

func distribute(ctx context.Context, hl plan.HostList, prog string, args []string) error {
	var ps []sch.Proc
	for _, h := range hl {
		proc := sch.Proc{
			Name:    h.PublicAddr,
			PubAddr: h.PublicAddr,
			Prog:    prog,
			Args:    args,
		}
		ps = append(ps, proc)
	}
	_, err := runner.RemoteRunAll(ctx, *user, ps, *verboseLog)
	return err
}
