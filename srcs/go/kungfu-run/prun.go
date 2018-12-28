package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	rch "github.com/luomai/kungfu/srcs/go/rchannel"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of tasks")
	hostList   = flag.String("H", rch.DefaultHostSpec().String(), "comma separated list of <hostname>:<nslots>[,<public addr>]")
	selfHost   = flag.String("self", "127.0.0.1", "")
	timeout    = flag.Duration("timeout", 10*time.Second, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
)

func init() {
	log.SetPrefix("[kungfu-run] ")
	flag.Parse()
	logArgs()
	logKungfuEnv()
}

func main() {
	hostSpecs, err := rch.ParseHostSpec(*hostList)
	if err != nil {
		exitErr(err)
	}
	specs, err := rch.GenClusterSpecs(*np, hostSpecs)
	if err != nil {
		exitErr(err)
	}
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		exitErr(errors.New("missing program name"))
	}
	prog := restArgs[0]
	args := restArgs[1:]

	var ps []*Proc
	for i, spec := range specs {
		if spec.Self.NetAddr.Host != *selfHost {
			continue
		}
		newValues := map[string]string{
			rch.ClusterSpecEnvKey:  spec.String(),
			`CUDA_VISIBLE_DEVICES`: strconv.Itoa(spec.Self.DeviceID),
			`PYTHONUNBUFFERED`:     `1`,
		}
		log.Printf("%s %q", prog, args)
		cmd := exec.Command(prog, args...)
		cmd.Env = updatedEnv(newValues)
		ps = append(ps, &Proc{
			name: fmt.Sprintf("%02d", i),
			cmd:  cmd,
		})
	}
	if len(ps) <= 0 {
		log.Print("No task to run on this node")
		return
	}
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	if err := runAll(ctx, ps, *verboseLog); err != nil {
		log.Print(err)
		if err != context.DeadlineExceeded {
			os.Exit(1)
		}
	}
}

func logArgs() {
	for i, a := range os.Args {
		log.Printf("args[%d]=%s", i, a)
	}
}

func logKungfuEnv() {
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, `KUNGFU_`) {
			log.Printf("env: %s", kv)
		}
	}
}

func updatedEnv(newValues map[string]string) []string {
	envMap := make(map[string]string)
	for _, kv := range os.Environ() {
		parts := strings.Split(kv, "=")
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
		}
	}
	for k, v := range newValues {
		envMap[k] = v
	}
	var envs []string
	for k, v := range envMap {
		envs = append(envs, fmt.Sprintf("%s=%s", k, v))
	}
	return envs
}

func exitErr(err error) {
	log.Printf("exit on error: %v", err)
	os.Exit(1)
}
