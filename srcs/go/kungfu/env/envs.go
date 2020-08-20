package env

// Internal environment variables set by kungfu-run, users should not set them.
const (
	ConfigServerEnvKey       = `KUNGFU_CONFIG_SERVER`
	InitClusterVersionEnvKey = `KUNGFU_INIT_CLUSTER_VERSION`
	ParentIDEnvKey           = `KUNGFU_PARENT_ID`

	PeerListEnvKey          = `KUNGFU_INIT_PEERS`
	RunnerListEnvKey        = `KUNGFU_INIT_RUNNERS`
	SelfSpecEnvKey          = `KUNGFU_SELF_SPEC` // self spec should never change during the life of a process
	AllReduceStrategyEnvKey = `KUNGFU_ALLREDUCE_STRATEGY`

	JobStartTimestamp  = `KUNGFU_JOB_START_TIMESTAMP`
	ProcStartTimestamp = `KUNGFU_PROC_START_TIMESTAMP`

	AllowNvLink = `KUNGFU_ALLOW_NVLINK`
)
