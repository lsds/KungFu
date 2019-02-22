package kungfuconfig

import (
	"fmt"
	"os"
	"time"
)

const (
	LogConfigVarsEnvKey    = `KUNGFU_CONFIG_LOG_CONFIG_VARS`
	RunWarmupEnvKey        = `KUNGFU_CONFIG_RUN_WARMUP`
	UseShmEnvKey           = `KUNGFU_CONFIG_USE_SHM`
	MonitoringPeriodEnvKey = `KUNGFU_CONFIG_MONITORING_PERIOD`
)

var ConfigEnvKeys = []string{
	LogConfigVarsEnvKey,
	RunWarmupEnvKey,
	UseShmEnvKey,
}

var (
	RunWarmup        = false
	UseShm           = false
	LogConfigVars    = false
	MonitoringPeriod time.Duration
)

func init() {
	if val := os.Getenv(RunWarmupEnvKey); len(val) > 0 {
		RunWarmup = isTrue(val)
	}
	if val := os.Getenv(UseShmEnvKey); len(val) > 0 {
		UseShm = isTrue(val)
	}
	if val := os.Getenv(LogConfigVarsEnvKey); len(val) > 0 {
		LogConfigVars = isTrue(val)
	}
	if val := os.Getenv(MonitoringPeriodEnvKey); len(val) > 0 {
		MonitoringPeriod = parseDuration(val)
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
		return 0
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
