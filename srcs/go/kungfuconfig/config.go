package kungfuconfig

import (
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/utils"
)

const (
	UseUnixSock = true
)

const (
	ConnRetryCount  = 200
	ConnRetryPeriod = 500 * time.Millisecond
)

const (
	EnableMonitoringEnvKey = `KUNGFU_CONFIG_ENABLE_MONITORING`
	MonitoringPeriodEnvKey = `KUNGFU_CONFIG_MONITORING_PERIOD`
	ShowDebugLogEnvKey     = `KUNGFU_CONFIG_SHOW_DEBUG_LOG`
	ConfigServerEnvKey     = `KUNGFU_CONFIG_SERVER`
	EnableAdaptiveEnvKey   = `KUNGFU_CONFIG_ENABLE_ADAPTIVE`
)

var ConfigEnvKeys = []string{
	EnableMonitoringEnvKey,
	MonitoringPeriodEnvKey,
	ShowDebugLogEnvKey,
	ConfigServerEnvKey,
	EnableAdaptiveEnvKey,
}

var (
	EnableMonitoring = false
	ShowDebugLog     = false
	EnableAdaptive   = false
	MonitoringPeriod = 1 * time.Second
)

func init() {
	if val := os.Getenv(EnableMonitoringEnvKey); len(val) > 0 {
		EnableMonitoring = isTrue(val)
	}
	if val := os.Getenv(MonitoringPeriodEnvKey); len(val) > 0 {
		MonitoringPeriod = parseDuration(val)
	}
	if val := os.Getenv(ShowDebugLogEnvKey); len(val) > 0 {
		ShowDebugLog = isTrue(val)
	}
	if val := os.Getenv(EnableAdaptiveEnvKey); len(val) > 0 {
		EnableAdaptive = isTrue(val)
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
