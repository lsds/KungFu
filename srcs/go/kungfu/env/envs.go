package env

// Internal environment variables set by kungfu-run, users should not set them.
const (
	ConfigServerEnvKey       = `KUNGFU_CONFIG_SERVER`
	InitClusterVersionEnvKey = `KUNGFU_INIT_CLUSTER_VERSION`
	ParentIDEnvKey           = `KUNGFU_PARENT_ID`
	HostListEnvKey           = `KUNGFU_HOST_LIST`

	PeerListEnvKey          = `KUNGFU_INIT_PEERS`
	HostSpecEnvKey          = `KUNGFU_HOST_SPEC`
	SelfSpecEnvKey          = `KUNGFU_SELF_SPEC` // self spec should never change during the life of a process
	AllReduceStrategyEnvKey = `KUNGFU_ALLREDUCE_STRATEGY`
)
