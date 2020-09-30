package app

import (
	"context"
	"fmt"
	"os"
	"path"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/job"
	"github.com/lsds/KungFu/srcs/go/kungfu/runner"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

func Main(args []string) {
	var f runner.FlagSet
	runner.Init(&f, args)
	if f.DelayStart > 0 {
		log.Warnf("delay start for %s", f.DelayStart)
		time.Sleep(f.DelayStart)
	}
	if f.BuiltinConfigPort > 0 {
		go runBuiltinConfigServer(f.BuiltinConfigPort)
	}
	if logfile := f.Logfile; len(logfile) > 0 {
		if len(f.LogDir) > 0 {
			logfile = path.Join(f.LogDir, logfile)
		}
		dir := path.Dir(logfile)
		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			log.Warnf("failed to create log dir %s: %v", dir, err)
		}
		lf, err := os.Create(logfile)
		if err != nil {
			utils.ExitErr(err)
		}
		defer lf.Close()
		log.SetOutput(lf)
	}
	t0 := time.Now()
	defer func(prog string) { log.Debugf("%s finished, took %s", prog, time.Since(t0)) }(utils.ProgName())
	localhostIPv4, err := runner.InferSelfIPv4(f.Self, f.NIC)
	if err != nil {
		utils.ExitErr(err)
	}
	log.Debugf("Using self=%s", plan.FormatIPv4(localhostIPv4))
	self := plan.PeerID{IPv4: localhostIPv4, Port: uint16(f.Port)}
	var hl plan.HostList
	var peers plan.PeerList
	var runners plan.PeerList
	// if len(f.HostList) > 0 {
	{
		hl = f.HostList
		// hl, err = runner.ResolveHostList(f.HostList, f.NIC)
		// if err != nil {
		// 	utils.ExitErr(fmt.Errorf("failed to parse -H: %v", err))
		// }
		runners = hl.GenRunnerList(uint16(f.Port)) // FIXME: assuming runner port is the same
		if _, ok := runners.Rank(self); !ok {
			utils.ExitErr(fmt.Errorf("%s not in %s", self, runners))
		}
		peers, err = hl.GenPeerList(f.ClusterSize, f.PortRange)
		if err != nil {
			utils.ExitErr(fmt.Errorf("failed to create peers: %v", err))
		}
	} //else {
	// peers, err = runner.ResolvePeerList(localhostIPv4, uint16(f.Port), f.PeerList)
	// if err != nil {
	// 	utils.ExitErr(fmt.Errorf("failed to resolve peers: %v", err))
	// }
	// log.Infof("-P resolved as %s", peers)
	// }
	j := job.Job{
		StartTime:   time.Unix(int64(f.JobStartTime), 0),
		Strategy:    f.Strategy,
		Parent:      self,
		HostList:    f.HostList,
		PortRange:   f.PortRange,
		Prog:        f.Prog,
		Args:        f.Args,
		LogDir:      f.LogDir,
		AllowNVLink: f.AllowNVLink,
	}
	ctx, cancel := context.WithCancel(context.Background())
	trap(cancel)
	if f.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, f.Timeout)
		defer cancel()
	}
	initCluster := plan.Cluster{
		Runners: runners,
		Workers: peers,
	}
	if f.Watch {
		ch := make(chan runner.Stage, 1)
		if f.InitVersion < 0 {
			log.Infof(xterm.Blue.S("waiting to be initialized"))
		} else {
			ch <- runner.Stage{
				Cluster: initCluster,
				Version: f.InitVersion,
			}
		}
		j.ConfigServer = f.ConfigServer
		runner.WatchRun(ctx, self, runners, ch, j, f.Keep, f.DebugPort)
	} else {
		runner.SimpleRun(ctx, localhostIPv4, initCluster, j, f.VerboseLog)
	}
}

func trap(cancel context.CancelFunc) {
	utils.Trap(func(sig os.Signal) {
		log.Warnf("%s trapped", sig)
		cancel()
		log.Debugf("cancelled")
	})
}
