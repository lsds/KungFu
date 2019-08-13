package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"time"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	runner "github.com/lsds/KungFu/srcs/go/runner/local"
	sch "github.com/lsds/KungFu/srcs/go/scheduler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	np         = flag.Int("np", runtime.NumCPU(), "number of peers")
	hostList   = flag.String("H", plan.DefaultHostSpec().String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	selfHost   = flag.String("self", "", "internal IP")
	timeout    = flag.Duration("timeout", 10*time.Second, "timeout")
	verboseLog = flag.Bool("v", true, "show task log")
	nicName    = flag.String("nic", "", "network interface name, for infer self IP")
	algo       = flag.String("algo", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.AllAlgoNames(), " | ")))

	configServerHost = flag.String("config-server-host", "127.0.0.1", "host of config server")
	configServerPort = flag.Int("config-server-port", 0, "will run config server on this port if not zero")
	watch            = flag.Bool("w", false, "watch config")
	watchPeriod      = flag.Duration("watch-period", 500*time.Millisecond, "")
	keep             = flag.Bool("k", false, "don't stop watch")
)

func init() {
	log.SetPrefix("[kungfu-prun] ")
	flag.Parse()
	utils.LogArgs()
	utils.LogKungfuEnv()
	utils.LogNICInfo()
	utils.LogCudaEnv()
	utils.LogNCCLEnv()
}

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
	log.Printf("Using selfHost=%s", selfIP)
	restArgs := flag.Args()
	if len(restArgs) < 1 {
		utils.ExitErr(errors.New("missing program name"))
	}
	prog := restArgs[0]
	args := restArgs[1:]

	hostSpecs, err := plan.ParseHostSpec(*hostList)
	if err != nil {
		utils.ExitErr(err)
	}

	jc := sch.JobConfig{
		PeerCount: *np,
		HostList:  *hostList,
		Prog:      prog,
		Args:      args,
	}

	useConfigServer := *configServerPort > 0

	var configServerAddr string
	if useConfigServer {
		configServerAddr = net.JoinHostPort(*configServerHost, strconv.Itoa(*configServerPort))
	}

	ps, cs, err := jc.CreateProcs(kb.ParseAlgo(*algo), configServerAddr)
	if err != nil {
		utils.ExitErr(err)
	}

	configClient, err := kf.NewConfigClient(configServerAddr)
	if err != nil {
		utils.ExitErr(err)
	}

	var updated chan string

	if useConfigServer {
		updated = make(chan string, 1)
		go watchConfigServer(configClient, updated)
		go runConfigServer(configServerAddr, nil)

		// updated = make(chan string, 1)
		// go runConfigServer(configServerAddr, updated)

		log.Printf("config server running at %s", configServerAddr)
		if err := initConfig(configClient, configServerAddr, hostSpecs, cs); err != nil {
			utils.ExitErr(err)
		}
	}

	if *watch {
		watchRun(configClient, selfIP, updated, prog, args, configServerAddr)
	} else {
		simpleRun(selfIP, ps, prog, args)
	}
}

func simpleRun(selfIP string, ps []sch.Proc, prog string, args []string) {
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

func runConfigServer(addr string, updated chan string) {
	server := http.Server{
		Addr:    addr,
		Handler: kf.NewConfigServer(updated),
	}
	server.ListenAndServe()
}

func initConfig(c *kf.ConfigClient, configServerAddr string, hostSpecs []plan.HostSpec, cs *plan.ClusterSpec) error {
	const initToken = "0"
	if err := c.PutConfig(initToken, kb.HostSpecEnvKey, hostSpecs); err != nil {
		return err
	}
	if err := c.PutConfig(initToken, kb.ClusterSpecEnvKey, cs); err != nil {
		return err
	}
	return nil
}
