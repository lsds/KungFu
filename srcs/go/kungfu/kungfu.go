package kungfu

import (
	"errors"
	"fmt"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Kungfu struct {
	sync.Mutex

	// immutable
	parent    plan.PeerID
	parents   plan.PeerList
	hostList  plan.HostList
	portRange plan.PortRange
	self      plan.PeerID
	strategy  kb.Strategy

	router *rch.Router
	server rch.Server

	// dynamic
	currentSession *session
	currentPeers   plan.PeerList
	checkpoint     string
	updated        bool
}

func New() (*Kungfu, error) {
	config, err := plan.ParseConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return NewFromConfig(config)
}

func NewFromConfig(config *plan.Config) (*Kungfu, error) {
	router := rch.NewRouter(config.Self)
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	return &Kungfu{
		parent:       config.Parent,
		parents:      config.Parents,
		currentPeers: config.InitPeers,
		self:         config.Self,
		hostList:     config.HostList,
		portRange:    config.PortRange,
		strategy:     config.Strategy,
		checkpoint:   config.Checkpoint,
		router:       router,
		server:       server,
	}, nil
}

func (kf *Kungfu) Start() int {
	go kf.server.Serve()
	if kc.EnableMonitoring {
		monitoringPort := kf.self.Port + 10000
		monitor.StartServer(int(monitoringPort))
		monitorAddr := plan.NetAddr{
			IPv4: kf.self.IPv4, // FIXME: use pubAddr
			Port: monitoringPort,
		}
		log.Infof("Kungfu peer %s started, monitoring endpoint http://%s/metrics", kf.self, monitorAddr)
	}
	kf.Update()
	return 0
}

func (kf *Kungfu) Close() int {
	if kc.EnableMonitoring {
		monitor.StopServer()
	}
	kf.server.Close() // TODO: check error
	return 0
}

var errSelfNotInCluster = errors.New("self not in cluster")

func (kf *Kungfu) CurrentSession() *session {
	kf.Lock()
	defer kf.Unlock()
	if kf.currentSession == nil {
		kf.updateTo(kf.currentPeers)
	}
	return kf.currentSession
}

func (kf *Kungfu) GetCheckpoint() string {
	return kf.checkpoint
}

func (kf *Kungfu) Update() bool {
	kf.Lock()
	defer kf.Unlock()
	return kf.updateTo(kf.currentPeers)
}

func (kf *Kungfu) updateTo(pl plan.PeerList) bool {
	if kf.updated {
		log.Debugf("ignore update")
		return true
	}
	log.Debugf("Kungfu::updateTo(%s), %d peers", pl, len(pl))
	kf.router.ResetConnections(pl)
	sess, exist := newSession(kf.strategy, kf.self, pl, kf.router)
	if !exist {
		return false
	}
	if err := sess.barrier(); err != nil {
		utils.ExitErr(fmt.Errorf("barrier failed after newSession: %v", err))
	}
	kf.currentSession = sess
	kf.updated = true
	return true
}

func (kf *Kungfu) Save(version, name string, buf *kb.Vector) error {
	return kf.router.P2P.SaveVersion(version, name, buf)
}

func par(ps plan.PeerList, f func(plan.PeerID) error) error {
	errs := make([]error, len(ps))
	var wg sync.WaitGroup
	for i, p := range ps {
		wg.Add(1)
		go func(i int, p plan.PeerID) {
			errs[i] = f(p)
			wg.Done()
		}(i, p)
	}
	wg.Wait()
	return mergeErrors(errs, "par")
}

func (kf *Kungfu) consensus(bs []byte) bool {
	n := len(bs)
	sess := kf.CurrentSession()
	{
		x := kb.NewVector(1, kb.I32)
		y := kb.NewVector(1, kb.I32)
		z := kb.NewVector(1, kb.I32)
		x.AsI32()[0] = int32(n)
		w1 := Workspace{SendBuf: x, RecvBuf: y, OP: kb.MIN, Name: ":consensus:len:min"}
		w2 := Workspace{SendBuf: x, RecvBuf: z, OP: kb.MAX, Name: ":consensus:len:max"}
		sess.AllReduce(w1)
		sess.AllReduce(w2)
		if !utils.BytesEq(x.Data, y.Data) || !utils.BytesEq(x.Data, z.Data) {
			return false
		}
	}
	if n == 0 {
		return true
	}
	{
		x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
		y := kb.NewVector(n, kb.U8)
		z := kb.NewVector(n, kb.U8)
		w1 := Workspace{SendBuf: x, RecvBuf: y, OP: kb.MIN, Name: ":consensus:min"}
		w2 := Workspace{SendBuf: x, RecvBuf: z, OP: kb.MAX, Name: ":consensus:max"}
		sess.AllReduce(w1)
		sess.AllReduce(w2)
		if !utils.BytesEq(x.Data, y.Data) || !utils.BytesEq(x.Data, z.Data) {
			return false
		}
	}
	return true
}

func (kf *Kungfu) propose(ckpt string, peers plan.PeerList) bool {
	if peers.Eq(kf.currentPeers) {
		log.Debugf("ingore unchanged proposal")
		return true
	}
	if digest := peers.Bytes(); !kf.consensus(digest) {
		log.Errorf("diverge proposal detected!")
		return true
	}
	{
		stage := run.Stage{Checkpoint: ckpt, Cluster: peers}
		if err := par(kf.parents, func(parent plan.PeerID) error {
			return kf.router.Send(parent.WithName("update"), stage.Encode(), rch.ConnControl, 0)
		}); err != nil {
			utils.ExitErr(err)
		}
	}
	func() {
		kf.Lock()
		defer kf.Unlock()
		kf.currentPeers = peers
		kf.checkpoint = ckpt
		kf.updated = false
	}()
	_, keep := peers.Lookup(kf.self)
	return keep
}

func (kf *Kungfu) ResizeCluster(ckpt string, newSize int) (bool, error) {
	log.Debugf("resize cluster to %d at %q", newSize, ckpt)
	peers, err := kf.hostList.GenPeerList(newSize, kf.portRange)
	if err != nil {
		return true, err
	}
	if keep := kf.propose(ckpt, peers); !keep {
		return false, nil
	}
	return kf.Update(), nil
}
