package runtime

import (
	"github.com/lsds/KungFu/srcs/go/plan"
)

type SystemParameters struct {
	User            string
	WorkerPortRange plan.PortRange
	RunnerPort      uint16
	HostList        plan.HostList
	ClusterSize     int
	Nic             string
}
