// Copied from prun.go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of peers")
	hostList   = flag.String("H", plan.DefaultHostSpec().String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	selfHost   = flag.String("self", "127.0.0.1", "internal IP")
	timeout    = flag.Duration("timeout", 10*time.Second, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	algo       = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.AllAlgoNames(), " | ")))
)

func init() {
	log.SetPrefix("[kungfu-prun] ")
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
}

func main() {
	log.Printf("Using selfHost=%s", *selfHost)
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	prog := restArgs[0]
	args := restArgs[1:]

	jc := sch.JobConfig{
		PeerCount: *np,
		HostList:  *hostList,
		Prog:      prog,
		Args:      args,
	}

	ps, cs, err := jc.CreateProcs(kb.ParseAlgo(*algo))
	if err != nil {
		utils.ExitErr(err)
	}
	myPs := sch.ForHost(*selfHost, ps)
	if len(myPs) <= 0 {
		log.Print("No task to run on this node")
		return
	}
	log.Printf("will parallel run %d instances of %s with %q", len(myPs), prog, args)

	go runConfigServer(cs)
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	d, err := utils.Measure(func() error { return runner.LocalRunAll(ctx, myPs, *verboseLog) })
	log.Printf("all %d/%d local peers finished, took %s", len(myPs), len(ps), d)
	if err != nil && err != context.DeadlineExceeded {
		utils.ExitErr(err)
	}
}

func runConfigServer(cs *plan.ClusterSpec) {
	configs := map[string]interface{}{
		kb.ClusterSpecEnvKey: cs,
	}
	server := http.Server{
		Addr: os.Getenv(kc.ConfigServerEnvKey),
		Handler: http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			log.Printf("%s %s %s", req.Method, req.URL.Path, req.URL.RawQuery)
			val := configs[req.FormValue("name")]
			json.NewEncoder(w).Encode(val)
		}),
	}
	server.ListenAndServe()
}
