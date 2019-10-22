package kungfurun

import (
	"errors"
	"flag"
	"fmt"
	"strings"
	"time"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func Init(f *FlagSet) {
	if err := f.Parse(); err != nil {
		utils.ExitErr(err)
	}
	if !f.Quiet {
		utils.LogArgs()
		utils.LogKungfuEnv()
		utils.LogNICInfo()
		utils.LogCudaEnv()
		utils.LogNCCLEnv()
	}
}

type FlagSet struct {
	ClusterSize int
	HostList    string
	PeerList    string

	User string

	portRange string
	PortRange plan.PortRange

	Self       string
	Timeout    time.Duration
	VerboseLog bool
	NIC        string

	strategy string
	Strategy kb.Strategy

	Port       int
	Watch      bool
	Checkpoint string

	Logfile string
	Quiet   bool

	Prog string
	Args []string
}

func (f *FlagSet) Register() {
	flag.IntVar(&f.ClusterSize, "np", 1, "number of peers")
	flag.StringVar(&f.HostList, "H", plan.DefaultHostList.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	flag.StringVar(&f.PeerList, "P", "", "comma separated list of <host>:<port>[:slot]")

	flag.StringVar(&f.User, "u", "", "user name for ssh")

	flag.StringVar(&f.portRange, "port-range", plan.DefaultPortRange.String(), "port range for the peers")

	flag.StringVar(&f.Self, "self", "", "internal IPv4")
	flag.DurationVar(&f.Timeout, "timeout", 0, "timeout")
	flag.BoolVar(&f.VerboseLog, "v", true, "show task log")
	flag.StringVar(&f.NIC, "nic", "", "network interface name, for infer self IP")
	flag.StringVar(&f.strategy, "strategy", "", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.StrategyNames(), " | ")))

	flag.IntVar(&f.Port, "port", 38080, "port for rchannel")
	flag.BoolVar(&f.Watch, "w", false, "watch config")
	flag.StringVar(&f.Checkpoint, "checkpoint", "0", "")

	flag.StringVar(&f.Logfile, "logfile", "", "path to log file")
	flag.BoolVar(&f.Quiet, "q", false, "don't log debug info")
}

var errMissingProgramName = errors.New("missing program name")

func (f *FlagSet) Parse() error {
	f.Register()
	flag.Parse()
	pr, err := plan.ParsePortRange(f.portRange)
	if err != nil {
		return fmt.Errorf("failed to parse -port-range: %v", err)
	}
	f.PortRange = *pr
	f.Strategy = kb.ParseStrategy(f.strategy)

	args := flag.Args()
	if len(args) < 1 {
		return errMissingProgramName
	}
	f.Prog = args[0]
	f.Args = args[1:]
	return nil
}
