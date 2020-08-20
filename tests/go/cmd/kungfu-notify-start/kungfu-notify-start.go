package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"net/http"

	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/kungfu/execution"
	"github.com/lsds/KungFu/srcs/go/kungfu/runner"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/client"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var errWaitPeerFailed = errors.New("wait peer failed")

var (
	configServer = flag.String("config-server", "", "config server URL")
)

func main() {
	flag.Parse()
	cluster, err := getClusterConfig(*configServer)
	if err != nil {
		utils.ExitErr(err)
	}
	notifyStart(*cluster)
}

func getClusterConfig(url string) (*plan.Cluster, error) {
	f, err := utils.OpenURL(url, http.DefaultClient, utils.ProgName())
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cluster plan.Cluster
	if err = json.NewDecoder(f).Decode(&cluster); err != nil {
		return nil, err
	}
	return &cluster, nil
}

func notifyStart(cluster plan.Cluster) {
	stage := runner.Stage{
		Version: 0,
		Cluster: cluster,
	}
	var self plan.PeerID
	client := client.New(self, config.UseUnixSock)
	var notify execution.PeerFunc = func(ctrl plan.PeerID) error {
		ctx, cancel := context.WithTimeout(context.TODO(), config.WaitRunnerTimeout)
		defer cancel()
		n, ok := client.Wait(ctx, ctrl)
		if !ok {
			return errWaitPeerFailed
		}
		if n > 0 {
			log.Warnf("%s is up after pinged %d times", ctrl, n+1)
		}
		return client.Send(ctrl.WithName("update"), stage.Encode(), connection.ConnControl, 0)
	}
	if err := notify.Par(cluster.Runners); err != nil {
		utils.ExitErr(err)
	}
}
