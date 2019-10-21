package kungfurun

import (
	"flag"
	"fmt"
	"runtime"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type FlagSet struct {
	ClusterSize int
	HostList    string
	PortRange   string
	Self        string
	Timeout     time.Duration
	VerboseLog  bool
	NIC         string
	Strategy    string

	Port       int
	Watch      bool
	Checkpoint string

	Logfile string
	Quiet   bool
}

func (f *FlagSet) Register() {
	flag.IntVar(&f.ClusterSize, "np", runtime.NumCPU(), "number of peers")
	flag.StringVar(&f.HostList, "H", plan.DefaultHostSpec.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")

	flag.StringVar(&f.PortRange, "port-range", plan.DefaultPortRange.String(), "port range for the peers")

	flag.StringVar(&f.Self, "self", "", "internal IPv4")
	flag.DurationVar(&f.Timeout, "timeout", 0, "timeout")
	flag.BoolVar(&f.VerboseLog, "v", true, "show task log")
	flag.StringVar(&f.NIC, "nic", "", "network interface name, for infer self IP")
	flag.StringVar(&f.Strategy, "strategy", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.StrategyNames(), " | ")))

	flag.IntVar(&f.Port, "port", 38080, "port for rchannel")
	flag.BoolVar(&f.Watch, "w", false, "watch config")
	flag.StringVar(&f.Checkpoint, "checkpoint", "0", "")

	flag.StringVar(&f.Logfile, "logfile", "", "path to log file")
	flag.BoolVar(&f.Quiet, "q", false, "don't log debug info")
}
