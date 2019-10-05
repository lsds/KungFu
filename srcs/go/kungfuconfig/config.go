package kungfuconfig

import (
	"fmt"
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/utils"
)

const (
	UseUnixSock = true
)

const (
	LogConfigVarsEnvKey    = `KUNGFU_CONFIG_LOG_CONFIG_VARS`
	RunWarmupEnvKey        = `KUNGFU_CONFIG_RUN_WARMUP`
	EnableMonitoringEnvKey = `KUNGFU_CONFIG_ENABLE_MONITORING`
	MonitoringPeriodEnvKey = `KUNGFU_CONFIG_MONITORING_PERIOD`
	ShowDebugLogEnvKey     = `KUNGFU_CONFIG_SHOW_DEBUG_LOG`
	ConfigServerEnvKey     = `KUNGFU_CONFIG_SERVER`
	EnableAdaptiveEnvKey   = `KUNGFU_CONFIG_ENABLE_ADAPTIVE`
)

var ConfigEnvKeys = []string{
	LogConfigVarsEnvKey,
	RunWarmupEnvKey,
	EnableMonitoringEnvKey,
	MonitoringPeriodEnvKey,
	ShowDebugLogEnvKey,
	ConfigServerEnvKey,
	EnableAdaptiveEnvKey,
}

var (
	RunWarmup        = false
	LogConfigVars    = false
	EnableMonitoring = false
	ShowDebugLog     = false
	EnableAdaptive   = false
	MonitoringPeriod = 1 * time.Second
)

func init() {
	if val := os.Getenv(RunWarmupEnvKey); len(val) > 0 {
		RunWarmup = isTrue(val)
	}
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
	if val := os.Getenv(LogConfigVarsEnvKey); len(val) > 0 {
		LogConfigVars = isTrue(val)
	}
	if LogConfigVars {
		logConfigVars()
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

func logConfigVars() {
	log("%s: %v", "RunWarmup", RunWarmup)
}

func log(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("[kungfu] %s\n", msg)
}
