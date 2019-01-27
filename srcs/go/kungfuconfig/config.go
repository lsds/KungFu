package kungfuconfig

import (
	"fmt"
	"os"
)

const (
	LogConfigVarsEnvKey    = `KUNGFU_CONFIG_LOG_CONFIG_VARS`
	RunWarmupEnvKey        = `KUNGFU_CONFIG_RUN_WARMUP`
	UseShmEnvKey           = `KUNGFU_CONFIG_USE_SHM`
	InplaceAllReduceEnvKey = `KUNGFU_CONFIG_INPLACE_ALLREDUCE`
)

var ConfigEnvKeys = []string{
	LogConfigVarsEnvKey,
	RunWarmupEnvKey,
	UseShmEnvKey,
	InplaceAllReduceEnvKey,
}

var (
	RunWarmup        = false
	UseShm           = false
	InplaceAllReduce = true
	LogConfigVars    = false
)

func init() {
	if val := os.Getenv(RunWarmupEnvKey); len(val) > 0 {
		RunWarmup = isTrue(val)
	}
	if val := os.Getenv(UseShmEnvKey); len(val) > 0 {
		UseShm = isTrue(val)
	}
	if val := os.Getenv(InplaceAllReduceEnvKey); len(val) > 0 {
		InplaceAllReduce = isTrue(val)
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

func logConfigVars() {
	log("%s: %v", "RunWarmup", RunWarmup)
	log("%s: %v", "UseShm", UseShm)
	log("%s: %v", "InplaceAllReduce", InplaceAllReduce)
}

func log(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("[kungfu] %s\n", msg)
}
