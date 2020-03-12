package main

import (
	"encoding/json"
	"os"

	"github.com/lsds/KungFu/srcs/go/plan"
)

func main() {
	pl := plan.PeerList{
		{
			IPv4: plan.MustParseIPv4(`127.0.0.1`),
			Port: plan.DefaultRunnerPort,
		},
	}
	hl := plan.HostList{
		{
			IPv4:  plan.MustParseIPv4(`127.0.0.1`),
			Slots: 4,
		},
	}
	ql, _ := hl.GenPeerList(4, plan.DefaultPortRange)
	c := plan.Cluster{
		Runners: pl,
		Workers: ql,
	}
	f, err := os.Create("init.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	json.NewEncoder(f).Encode(&c)
}
