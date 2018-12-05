package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	rch "github.com/luomai/kungfu/srcs/go/rchannel"
	"github.com/luomai/kungfu/srcs/go/xterm"
)

var (
	np       = flag.Int("np", runtime.NumCPU(), "number of tasks")
	m        = flag.Int("m", 4, "number of GPUs per host")
	hostList = flag.String("hosts", "127.0.0.1", "comma separated list of hosts")
	selfHost = flag.String("self", "127.0.0.1", "")
	timeout  = flag.Duration("timeout", 10*time.Second, "timeout")
)

var (
	basicColors = []xterm.Color{
		xterm.Green,
		xterm.Yellow,
	}

	waiting []bool
	lock    sync.Mutex
)

func main() {
	flag.Parse()
	logArgs()
	hosts := strings.Split(*hostList, ",")
	specs := rch.GenCluster(*np, hosts, *m)
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		log.Print("missing program name")
		os.Exit(1)
	}
	prog := restArgs[0]
	args := restArgs[1:]

	var ps []*Proc
	for _, spec := range specs {
		if spec.Self.NetAddr.Host != *selfHost {
			continue
		}
		envs := []string{
			// FIXME: passdown more envs
			fmt.Sprintf(`%s=%s`, `PATH`, os.Getenv(`PATH`)),
			fmt.Sprintf(`%s=%s`, `HOME`, os.Getenv(`HOME`)),
			fmt.Sprintf(`%s=%s`, rch.ClusterSpecEnvKey, spec),
			fmt.Sprintf(`%s=%d`, `CUDA_VISIBLE_DEVICES`, spec.Self.DeviceID),
			fmt.Sprintf(`%s=%s`, `PYTHONUNBUFFERED`, `1`),
		}
		log.Printf("%s %q", prog, args)
		cmd := exec.Command(prog, args...)
		cmd.Env = envs
		ps = append(ps, &Proc{
			name: fmt.Sprintf("%02d", spec.MyRank()),
			cmd:  cmd,
		})
	}

	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	if err := runAll(ctx, ps); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func logArgs() {
	for i, a := range os.Args {
		fmt.Printf("args[%d]=%s\n", i, a)
	}
}
