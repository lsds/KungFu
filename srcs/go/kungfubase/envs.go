package kungfubase

const (
	// FIXME: merge with config.go
	CheckpointEnvKey = `KUNGFU_INIT_CKPT`
	ParentIDEnvKey   = `KUNGFU_PARENT_ID`
	HostListEnvKey   = `KUNGFU_HOST_LIST`

	PeerListEnvKey      = `KUNGFU_INIT_PEERS`
	HostSpecEnvKey      = `KUNGFU_HOST_SPEC`
	SelfSpecEnvKey      = `KUNGFU_SELF_SPEC` // self spec should never change during the life of a process
	InitSessEnvKey      = `KUNGFU_INIT_SESS`
	InitStepEnvKey      = `KUNGFU_INIT_STEP`
	AllReduceStrategyEnvKey = `KUNGFU_ALLREDUCE_ALGO` // FIXME: remove it
)
