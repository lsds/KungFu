package kungfu

import (
	"fmt"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

func timeoutHelper(timeoutDuration time.Duration, op func(), timeoutCallback func()) {
	ch := make(chan bool, 1)
	go func() {
		op()
		ch <- true
	}()
	select {
	case <-ch:
	case <-time.After(timeoutDuration):
		timeoutCallback()
	}
}

func healthCeck(self plan.PeerID, target plan.PeerID) {
	conn := rch.NewPingConnection(plan.NetAddr(self), plan.NetAddr(target))
	defer conn.Close()
	var empty rch.Message
	err := conn.Send("ping", empty, rch.NoFlag)
	if err != nil {
		log.Errorf("ping failed %s -> %s", plan.NetAddr(self), plan.NetAddr(target))
		// TODO report sess.peers[rank] as gone
	} else {
		log.Errorf("successful ping %s -> %s", plan.NetAddr(self), plan.NetAddr(target))
		fmt.Println("")
	}
}
