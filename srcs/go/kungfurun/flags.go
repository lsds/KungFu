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
	ConfigServer string
	ClusterSize  int
	HostList     string
	PeerList     string

	User string

	PortRange plan.PortRange

	Self        string
	Timeout     time.Duration
	VerboseLog  bool
	NIC         string
	AllowNVLink bool

	Strategy kb.Strategy

	Port        int
	Watch       bool
	InitVersion int

	Logfile string
	LogDir  string
	Quiet   bool

	Prog string
	Args []string
}

func (f *FlagSet) Register(flag *flag.FlagSet) {
	flag.IntVar(&f.ClusterSize, "np", 1, "number of peers")
	flag.StringVar(&f.HostList, "H", plan.DefaultHostList.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	flag.StringVar(&f.PeerList, "P", "", "comma separated list of <host>:<port>[:slot]")

	flag.StringVar(&f.User, "u", "", "user name for ssh")

	f.PortRange = plan.DefaultPortRange
	flag.Var(&f.PortRange, "port-range", "port range for the peers")

	flag.StringVar(&f.Self, "self", "", "internal IPv4")
	flag.DurationVar(&f.Timeout, "timeout", 0, "timeout")
	flag.BoolVar(&f.VerboseLog, "v", true, "show task log")
	flag.StringVar(&f.NIC, "nic", "", "network interface name, for infer self IP")
	flag.BoolVar(&f.AllowNVLink, "allow-nvlink", false, "allow NCCL to discover NVLink")

	f.Strategy = kb.DefaultStrategy
	flag.Var(&f.Strategy, "strategy", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(kb.StrategyNames(), " | ")))

	flag.IntVar(&f.Port, "port", 38080, "port for rchannel")
	flag.BoolVar(&f.Watch, "w", false, "watch config")
	flag.IntVar(&f.InitVersion, "init-version", 0, "initial cluster version")
	flag.StringVar(&f.ConfigServer, "config-server", "", "config server URL")

	flag.StringVar(&f.Logfile, "logfile", "", "path to log file")
	flag.StringVar(&f.LogDir, "logdir", "", "path to log dir")
	flag.BoolVar(&f.Quiet, "q", false, "don't log debug info")
}

var errMissingProgramName = errors.New("missing program name")

func (f *FlagSet) Parse() error {
	f.Register(flag.CommandLine)
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		return errMissingProgramName
	}
	f.Prog = args[0]
	f.Args = args[1:]
	return nil
}
