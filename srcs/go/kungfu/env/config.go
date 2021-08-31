package env

import (
	"fmt"
	"os"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type ElasticMode int

var (
	ElasticModeDefault ElasticMode = 0
	ElasticModeReload  ElasticMode = 1
)

// Set implements flags.Value::Set
func (e *ElasticMode) Set(val string) error {
	value, err := parseElasticMode(val)
	if err != nil {
		return err
	}
	*e = *value
	return nil
}

var elasticModeNames = map[ElasticMode]string{
	ElasticModeDefault: "",
	ElasticModeReload:  "reload",
}

func (e ElasticMode) String() string {
	return elasticModeNames[e]
}

type Config struct {
	ConfigServer string
	Parent       plan.PeerID
	InitRunners  plan.PeerList
	Self         plan.PeerID
	Strategy     kb.Strategy

	InitClusterVersion string
	InitProgress       uint64
	InitPeers          plan.PeerList

	Single      bool
	ElasticMode ElasticMode
}

func ParseConfigFromEnv() (*Config, error) {
	if _, ok := os.LookupEnv(SelfSpecEnvKey); !ok {
		return singleProcessEnv(), nil
	}
	self, err := getSelfFromEnv()
	if err != nil {
		return nil, err
	}
	parent, err := getParentFromEnv()
	if err != nil {
		return nil, err
	}
	initRunners, err := getInitRunnersFromEnv()
	if err != nil {
		return nil, err
	}
	initPeers, err := getInitPeersFromEnv()
	if err != nil {
		return nil, err
	}
	strategy, err := kb.ParseStrategy(os.Getenv(AllReduceStrategyEnvKey))
	if err != nil {
		return nil, err
	}
	elasticMode, err := parseElasticMode(os.Getenv(ElasticModeEnvKey))
	if err != nil {
		return nil, err
	}
	initProgress, err := parseInitProgress(os.Getenv(InitProgressEnvKey))
	if err != nil {
		return nil, err
	}
	return &Config{
		ConfigServer:       getConfigServerFromEnv(),
		Self:               *self,
		Parent:             *parent,
		InitRunners:        initRunners,
		InitPeers:          initPeers,
		Strategy:           *strategy,
		InitClusterVersion: os.Getenv(InitClusterVersionEnvKey),
		ElasticMode:        *elasticMode,
		InitProgress:       initProgress,
	}, nil
}

func parseElasticMode(val string) (*ElasticMode, error) {
	if val == "" {
		return &ElasticModeDefault, nil
	}
	if val == "reload" {
		return &ElasticModeReload, nil
	}
	return nil, fmt.Errorf("invalid %s: %q", ElasticModeEnvKey, val)
}

func parseInitProgress(val string) (uint64, error) {
	if val == "" {
		return 0, nil
	}
	n, err := strconv.ParseInt(val, 10, 64)
	if err != nil {
		return 0, err
	}
	return uint64(n), nil
}

func SingleMachineEnv(rank, size int) (*Config, error) {
	pl, err := plan.DefaultHostList.GenPeerList(size, plan.DefaultPortRange)
	if err != nil {
		return nil, err
	}
	return &Config{
		Self:      pl[rank],
		InitPeers: pl,
		Strategy:  kb.DefaultStrategy,
	}, nil
}

func singleProcessEnv() *Config {
	pl, _ := plan.DefaultHostList.GenPeerList(1, plan.DefaultPortRange)
	self := pl[0]
	return &Config{
		Self:      self,
		InitPeers: plan.PeerList{self},
		Strategy:  kb.DefaultStrategy,
		Single:    true,
	}
}

func getConfigServerFromEnv() string {
	return os.Getenv(ConfigServerEnvKey)
}

func getSelfFromEnv() (*plan.PeerID, error) {
	config, ok := os.LookupEnv(SelfSpecEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", SelfSpecEnvKey)
	}
	return plan.ParsePeerID(config)
}

func getParentFromEnv() (*plan.PeerID, error) {
	val, ok := os.LookupEnv(ParentIDEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", ParentIDEnvKey)
	}
	return plan.ParsePeerID(val)
}

func getInitPeersFromEnv() (plan.PeerList, error) {
	val, ok := os.LookupEnv(PeerListEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", PeerListEnvKey)
	}
	return plan.ParsePeerList(val)
}

func getInitRunnersFromEnv() (plan.PeerList, error) {
	val, ok := os.LookupEnv(RunnerListEnvKey)
	if !ok {
		return nil, fmt.Errorf("%s not set", RunnerListEnvKey)
	}
	return plan.ParsePeerList(val)
}
