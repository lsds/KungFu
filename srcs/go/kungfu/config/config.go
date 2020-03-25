package config

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
	EnableMonitoringEnvKey     = `KUNGFU_CONFIG_ENABLE_MONITORING`
	EnableStallDetectionEnvKey = `KUNGFU_CONFIG_ENABLE_STALL_DETECTION`
	LogLevelEnvKey             = `KUNGFU_CONFIG_LOG_LEVEL`
	MonitoringPeriodEnvKey     = `KUNGFU_CONFIG_MONITORING_PERIOD`
	StrategyHashMethodEnvKey   = `KUNGFU_CONFIG_STRATEGY_HASH_METHOD`
)

var ConfigEnvKeys = []string{
	EnableMonitoringEnvKey,
	MonitoringPeriodEnvKey,
	LogLevelEnvKey,
	StrategyHashMethodEnvKey,
}

var (
	EnableMonitoring     = false
	EnableStallDetection = false
	LogLevel             = `INFO`
	MonitoringPeriod     = 1 * time.Second
	StrategyHashMethod   = `NAME`
)

func init() {
	if val := os.Getenv(EnableMonitoringEnvKey); len(val) > 0 {
		EnableMonitoring = isTrue(val)
	}
	if val := os.Getenv(EnableStallDetectionEnvKey); len(val) > 0 {
		EnableStallDetection = isTrue(val)
	}
	if val := os.Getenv(MonitoringPeriodEnvKey); len(val) > 0 {
		MonitoringPeriod = parseDuration(val)
	}
	if val := os.Getenv(LogLevelEnvKey); len(val) > 0 {
		LogLevel = strings.ToUpper(val) // FIXME: check enum value
	}
	if val := os.Getenv(StrategyHashMethodEnvKey); len(val) > 0 {
		StrategyHashMethod = strings.ToUpper(val) // FIXME: check enum value
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
