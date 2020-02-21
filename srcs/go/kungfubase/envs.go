package kungfubase

const (
	// FIXME: merge with config.go
	ConfigServerEnvKey = `KUNGFU_CONFIG_SERVER`
	InitStepEnvKey     = `KUNGFU_INIT_STEP`
	ParentIDEnvKey     = `KUNGFU_PARENT_ID`
	HostListEnvKey     = `KUNGFU_HOST_LIST`
	PortRangeEnvKey    = `KUNGFU_PORT_RANGE`

	PeerListEnvKey          = `KUNGFU_INIT_PEERS`
	HostSpecEnvKey          = `KUNGFU_HOST_SPEC`
	SelfSpecEnvKey          = `KUNGFU_SELF_SPEC` // self spec should never change during the life of a process
	AllReduceStrategyEnvKey = `KUNGFU_ALLREDUCE_STRATEGY`
)
