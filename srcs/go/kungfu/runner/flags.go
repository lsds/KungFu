package runner

import (
	"errors"
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/hostfile"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func Init(f *FlagSet, args []string) {
	if err := f.Parse(args); err != nil {
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
	hostList     string
	hostFile     string
	HostList     plan.HostList
	peerList     string

	User string

	PortRange plan.PortRange

	Self        string
	Timeout     time.Duration
	VerboseLog  bool
	NIC         string
	AllowNVLink bool

	Strategy base.Strategy

	Port        int
	DebugPort   int
	Watch       bool
	Keep        bool
	InitVersion int

	Logfile string
	LogDir  string
	Quiet   bool

	JobStartTime int
	Prog         string
	Args         []string

	// debug and testing flags
	BuiltinConfigPort int
	DelayStart        time.Duration
}

func (f *FlagSet) Register(flag *flag.FlagSet) {
	flag.IntVar(&f.ClusterSize, "np", 1, "number of peers")
	flag.StringVar(&f.hostList, "H", plan.DefaultHostList.String(), "comma separated list of <internal IP>:<nslots>[:<public addr>]")
	flag.StringVar(&f.hostFile, "hostfile", "", "path to hostfile, will override -H if specified")
	flag.StringVar(&f.peerList, "P", "", "comma separated list of <host>:<port>[:slot]")

	flag.StringVar(&f.User, "u", "", "user name for ssh")

	f.PortRange = plan.DefaultPortRange
	flag.Var(&f.PortRange, "port-range", "port range for the peers")

	flag.StringVar(&f.Self, "self", "", "internal IPv4")
	flag.DurationVar(&f.Timeout, "timeout", 0, "timeout")
	flag.BoolVar(&f.VerboseLog, "v", true, "show task log")
	flag.StringVar(&f.NIC, "nic", "", "network interface name, for infer self IP")
	flag.BoolVar(&f.AllowNVLink, "allow-nvlink", false, "allow NCCL to discover NVLink")

	f.Strategy = base.DefaultStrategy
	flag.Var(&f.Strategy, "strategy", fmt.Sprintf("all reduce strategy, options are: %s", strings.Join(base.StrategyNames(), " | ")))

	flag.IntVar(&f.Port, "port", int(plan.DefaultRunnerPort), "port for rchannel")
	flag.IntVar(&f.DebugPort, "debug-port", 0, "port for HTTP debug server")
	flag.BoolVar(&f.Watch, "w", false, "watch config")
	flag.BoolVar(&f.Keep, "k", false, "stay alive after works finished")
	flag.IntVar(&f.InitVersion, "init-version", 0, "initial cluster version")
	flag.StringVar(&f.ConfigServer, "config-server", "", "config server URL")

	flag.IntVar(&f.JobStartTime, "t0", int(time.Now().Unix()), "job start timestamp")
	flag.StringVar(&f.Logfile, "logfile", "", "path to log file")
	flag.StringVar(&f.LogDir, "logdir", "", "path to log dir")
	flag.BoolVar(&f.Quiet, "q", false, "don't log debug info")

	flag.DurationVar(&f.DelayStart, "delay", 0, "delay start for testing purpose")
	flag.IntVar(&f.BuiltinConfigPort, "builtin-config-port", 0, "will run a builtin config server if not zero")
}

var errMissingProgramName = errors.New("missing program name")

func (f *FlagSet) Parse(args []string) error {
	commandLine := flag.NewFlagSet(args[0], flag.ExitOnError)
	f.Register(commandLine)
	commandLine.Parse(args[1:])
	if err := f.resolveHostList(); err != nil {
		return err
	}
	args = commandLine.Args()
	if len(args) < 1 {
		return errMissingProgramName
	}
	f.Prog = args[0]
	f.Args = args[1:]
	return nil
}

func (f *FlagSet) resolveHostList() error {
	if len(f.hostFile) > 0 {
		hl, err := hostfile.ParseFile(f.hostFile)
		if err != nil {
			return err
		}
		f.HostList = hl
	} else {
		hl, err := plan.ParseHostList(f.hostList)
		if err != nil {
			return err
		}
		f.HostList = hl
	}
	return nil
}
