package kungfuconfig

import (
	"os"
	"strings"
	"time"

	"github.com/lsds/KungFu/srcs/go/utils"
)

const (
	UseUnixSock = true
)

const (
	ConnRetryCount  = 500
	ConnRetryPeriod = 200 * time.Millisecond
)

const (
	EnableMonitoringEnvKey = `KUNGFU_CONFIG_ENABLE_MONITORING`
	MonitoringPeriodEnvKey = `KUNGFU_CONFIG_MONITORING_PERIOD`
	LogLevelEnvKey         = `KUNGFU_CONFIG_LOG_LEVEL`
	ShardHashMethodEnvKey  = `KUNGFU_CONFIG_SHARD_HASH_METHOD`
)

var ConfigEnvKeys = []string{
	EnableMonitoringEnvKey,
	MonitoringPeriodEnvKey,
	LogLevelEnvKey,
	ShardHashMethodEnvKey,
}

var (
	EnableMonitoring = false
	LogLevel         = `INFO`
	MonitoringPeriod = 1 * time.Second
	ShardHashMethod  = `ID`
)

func init() {
	if val := os.Getenv(EnableMonitoringEnvKey); len(val) > 0 {
		EnableMonitoring = isTrue(val)
	}
	if val := os.Getenv(MonitoringPeriodEnvKey); len(val) > 0 {
		MonitoringPeriod = parseDuration(val)
	}
	if val := os.Getenv(LogLevelEnvKey); len(val) > 0 {
		LogLevel = strings.ToUpper(val)
	}
	if val := os.Getenv(ShardHashMethodEnvKey); len(val) > 0 {
		ShardHashMethod = strings.ToUpper(val)
	}
}

func isTrue(val string) bool {
	return val == "true"
}

func parseDuration(val string) time.Duration {
	d, err := time.ParseDuration(val)
	if err != nil {
		utils.ExitErr(err)
	}
	return d
}
