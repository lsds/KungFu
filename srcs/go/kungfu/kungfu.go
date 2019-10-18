package kungfu

import (
	"errors"
	"fmt"
	"os"
	"sync"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	run "github.com/lsds/KungFu/srcs/go/kungfurun"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/monitor"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/store"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Config struct {
	Strategy kb.Strategy
}

type Kungfu struct {
	sync.Mutex

	// immutable
	parent    plan.PeerID
	parents   plan.PeerList
	hostList  plan.HostList
	portRange plan.PortRange
	self      plan.PeerID

	store  *store.VersionedStore
	router *rch.Router
	server rch.Server
	config Config

	// dynamic
	currentSession *session
	currentPeers   plan.PeerList
	checkpoint     string
	updated        bool
}

func getParentIDs(hl plan.HostList, parent plan.PeerID) plan.PeerList {
	var ps plan.PeerList
	for _, h := range hl {
		ps = append(ps, plan.PeerID{IPv4: h.IPv4, Port: parent.Port})
	}
	return ps
}

func New(config Config) (*Kungfu, error) {
	env, err := plan.ParseEnv()
	if err != nil {
		return nil, err
	}
	store := store.NewVersionedStore(3)
	router := rch.NewRouter(env.Self, store)
	server, err := rch.NewServer(router)
	if err != nil {
		return nil, err
	}
	return &Kungfu{
		parent:       env.Parent,
		parents:      getParentIDs(env.HostList, env.Parent),
		currentPeers: env.InitPeers,
		self:         env.Self,
		hostList:     env.HostList,
		portRange:    env.PortRange,
		checkpoint:   os.Getenv(kb.CheckpointEnvKey),
		store:        store,
		router:       router,
		server:       server,
		config:       config,
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
	sess, exist := newSession(kf.config, kf.self, pl, kf.router)
	if !exist {
		return false
	}
	if err := sess.Barrier(); err != nil {
		utils.ExitErr(fmt.Errorf("Barrier failed after newSession: %v", err))
	}
	kf.currentSession = sess
	kf.updated = true
	return true
}

func (kf *Kungfu) Save(version, name string, buf *kb.Vector) error {
	blob := &store.Blob{Data: buf.Data}
	return kf.store.Create(version, name, blob)
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
		if !bytesEq(x.Data, y.Data) || !bytesEq(x.Data, z.Data) {
			return false
		}
	}
	{
		x := &kb.Vector{Data: bs, Count: n, Type: kb.U8}
		y := kb.NewVector(n, kb.U8)
		z := kb.NewVector(n, kb.U8)
		w1 := Workspace{SendBuf: x, RecvBuf: y, OP: kb.MIN, Name: ":consensus:min"}
		w2 := Workspace{SendBuf: x, RecvBuf: z, OP: kb.MAX, Name: ":consensus:max"}
		sess.AllReduce(w1)
		sess.AllReduce(w2)
		if !bytesEq(x.Data, y.Data) || !bytesEq(x.Data, z.Data) {
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

func bytesEq(x, y []byte) bool {
	if len(x) != len(y) {
		return false
	}
	for i, a := range x {
		if a != y[i] {
			return false
		}
	}
	return true
}
