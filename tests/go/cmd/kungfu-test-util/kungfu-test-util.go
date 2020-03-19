package main

import (
	"flag"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var (
	kill = flag.String("kill", "", "peer id to terminate")
)

func main() {
	flag.Parse()
	switch {
	case len(*kill) > 0:
		target, err := plan.ParsePeerID(*kill)
		if err != nil {
			utils.ExitErr(err)
		}
		terminate(*target)
		return
	default:
		flag.Usage()
		utils.ShowBuildInfo()
	}
}

func terminate(target plan.PeerID) {
	router := rch.NewRouter(plan.PeerID{}) // FIXME: don't retry connect
	if err := router.Send(target.WithName("exit"), nil, rch.ConnControl, rch.NoFlag); err != nil {
		log.Errorf("failed to send exit signal sent to %s", target)
		return
	}
	log.Infof("exit signal sent to %s", target)
}
