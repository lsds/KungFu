package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/runner"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/proc"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/runner/remote"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

var (
	hostList   = flag.String("H", plan.DefaultHostList.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	timeout    = flag.Duration("timeout", 0, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	user       = flag.String("u", "", "user name for ssh")
	quiet      = flag.Bool("q", false, "don't log debug info")
	logDir     = flag.String("logdir", ".", "")
)

func init() {
	log.SetFlags(0)
	flag.Parse()
	if !*quiet {
		utils.LogArgs()
		utils.LogKungfuEnv()
		utils.LogNICInfo()
		utils.LogCudaEnv()
		utils.LogNCCLEnv()
	}
}

func progName() string {
	if len(os.Args) > 0 {
		return path.Base(os.Args[0])
	}
	return ""
}

func main() {
	args := flag.Args()
	if len(args) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	t0 := time.Now()
	defer func(prog string) {
		highlight := xterm.Yellow.S("`") + xterm.Blue.S(strings.Join(args, " ")) + xterm.Yellow.S("`")
		log.Infof("%s %s took %s", prog, highlight, time.Since(t0))
	}(progName())
	hl, err := runner.ParseHostList(*hostList)
	if err != nil {
		utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
	}
	log.Infof("Using %d hosts", len(hl))
	ctx, cancel := context.WithCancel(context.Background())
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	if err := distribute(ctx, hl, args[0], args[1:]); err != nil {
		log.Errorf("%v", err)
		utils.ExitErr(err)
	}
}

func distribute(ctx context.Context, hl []runner.HostSpec, prog string, args []string) error {
	var ps []proc.Proc
	for _, h := range hl {
		proc := proc.Proc{
			Name:     h.PublicAddr,
			Hostname: h.PublicAddr,
			Prog:     prog,
			Args:     args,
		}
		ps = append(ps, proc)
	}
	return remote.RemoteRunAll(ctx, *user, ps, *verboseLog, *logDir)
}
