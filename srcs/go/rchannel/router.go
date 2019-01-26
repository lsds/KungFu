package rchannel

import (
	"expvar"
	"net"
	"os"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/metrics"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/shm"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Router struct {
	localAddr  plan.NetAddr
	bufferPool *BufferPool
	connPool   *ConnectionPool

	totalMsgSent   *expvar.Int
	totalMsgRecv   *expvar.Int
	totalBytesSent *expvar.Int
	totalBytesRecv *expvar.Int
	sendRate       *expvar.Float
	recvRate       *expvar.Float
}

func NewRouter(self plan.PeerSpec) *Router {
	return &Router{
		localAddr:  self.NetAddr,
		bufferPool: newBufferPool(),     // in-comming messages
		connPool:   newConnectionPool(), // out-going connections

		totalMsgSent:   expvar.NewInt("total_msg_sent"),
		totalMsgRecv:   expvar.NewInt("total_msg_recv"),
		totalBytesSent: expvar.NewInt("total_bytes_sent"),
		totalBytesRecv: expvar.NewInt("total_bytes_recv"),
		sendRate:       expvar.NewFloat("send_bytes_per_sec"),
		recvRate:       expvar.NewFloat("recv_bytes_per_sec"),
	}
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a plan.Addr) (*Channel, error) {
	conn, err := r.connPool.get(a.NetAddr(), r.localAddr)
	if err != nil {
		return nil, err
	}
	return newChannel(a.Name, conn), nil
}

// Send sends data in buf to given Addr
func (r *Router) Send(a plan.Addr, buf []byte) error {
	m := Message{
		Length: uint32(len(buf)),
		Data:   buf,
	}
	if err := r.send(a, m); err != nil {
		log.Errorf("Router::Send failed: %v", err)
		// TODO: retry
		os.Exit(1)
		// return err
	}
	return nil
}

func (r *Router) send(a plan.Addr, m Message) error {
	// log.Infof("%s::%s", "Router", "Send")
	ch, err := r.getChannel(a)
	if err != nil {
		return err
	}
	if err := ch.Send(m); err != nil {
		return err
	}
	r.totalMsgSent.Add(1)
	r.totalBytesSent.Add(int64(m.Length))
	return nil
}

// Recv recevies a message from given Addr
func (r *Router) Recv(a plan.Addr) Message {
	// log.Infof("%s::%s(%s)", "Router", "Recv", a)
	// TODO: reduce memory copy
	m := *<-r.bufferPool.require(a)
	r.totalMsgRecv.Add(1)
	r.totalBytesRecv.Add(int64(m.Length))
	// TODO: add timeout
	return m
}

func (r *Router) acceptOne(conn net.Conn, shm shm.Shm) (string, *Message, error) {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return "", nil, err
	}
	var m Message
	if mh.BodyInShm != 0 {
		var mt messageTail
		if err := mt.ReadFrom(conn); err != nil {
			return "", nil, err
		}
		m.Length = mt.Length
		m.Data = make([]byte, m.Length)
		shm.Seek(int(mt.Offset))
		shm.Read(m.Data, int(m.Length))
		mt.WriteTo(conn)
	} else {
		if err := m.ReadFrom(conn); err != nil {
			return "", nil, err
		}
	}
	return string(mh.Name), &m, nil
}

var newShm = shm.New

func (r *Router) stream(conn net.Conn, remote plan.NetAddr) (int, error) {
	var shm shm.Shm
	if kc.UseShm && remote.Host == r.localAddr.Host {
		var err error
		if shm, err = newShm(plan.ShmNameFor(remote, r.localAddr)); err != nil {
			return 0, err
		}
		defer shm.Close()
	}
	for i := 0; ; i++ {
		name, m, err := r.acceptOne(conn, shm)
		if err != nil {
			return i, err
		}
		r.bufferPool.require(remote.WithName(name)) <- m
	}
}

func (r *Router) UpdateRate() {
	t0 := time.Unix(0, metrics.StartTime.Value())
	d := time.Since(t0)

	rx := r.totalBytesRecv.Value()
	tx := r.totalBytesSent.Value()

	rxRate := utils.Rate(rx, d)
	txRate := utils.Rate(tx, d)
	r.recvRate.Set(rxRate)
	r.sendRate.Set(txRate)

	const Mi = 1 << 20
	log.Infof("rx_rate: %s, tx_rate: %s", utils.ShowRate(rxRate), utils.ShowRate(txRate))
}
