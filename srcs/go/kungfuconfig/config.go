package kungfuconfig

import (
	"fmt"
	"os"
	"time"

	"github.com/lsds/KungFu/srcs/go/utils"
)

const (
	LogConfigVarsEnvKey    = `KUNGFU_CONFIG_LOG_CONFIG_VARS`
	RunWarmupEnvKey        = `KUNGFU_CONFIG_RUN_WARMUP`
	UseShmEnvKey           = `KUNGFU_CONFIG_USE_SHM`
	EnableMonitoringEnvKey = `KUNGFU_CONFIG_ENABLE_MONITORING`
	MonitoringPeriodEnvKey = `KUNGFU_CONFIG_MONITORING_PERIOD`
	ShowDebugLogEnvKey     = `KUNGFU_CONFIG_SHOW_DEBUG_LOG`
	ConfigServerEnvKey     = `KUNGFU_CONFIG_SERVER`
)

var ConfigEnvKeys = []string{
	LogConfigVarsEnvKey,
	RunWarmupEnvKey,
	UseShmEnvKey,
	EnableMonitoringEnvKey,
	MonitoringPeriodEnvKey,
	ShowDebugLogEnvKey,
	ConfigServerEnvKey,
}

var (
	RunWarmup        = false
	UseShm           = false
	LogConfigVars    = false
	EnableMonitoring = false
	ShowDebugLog     = false
	MonitoringPeriod = 1 * time.Second
)

func init() {
	if val := os.Getenv(RunWarmupEnvKey); len(val) > 0 {
		RunWarmup = isTrue(val)
	}
	if val := os.Getenv(UseShmEnvKey); len(val) > 0 {
		UseShm = isTrue(val)
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
	log("%s: %v", "UseShm", UseShm)
}

func log(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("[kungfu] %s\n", msg)
}
