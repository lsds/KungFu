package rchannel

import (
	"expvar"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/metrics"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Router struct {
	localHost  string
	localPort  uint32
	localSock  string
	bufferPool *BufferPool
	connPool   *ConnectionPool

	totalMsgSent   *expvar.Int
	totalMsgRecv   *expvar.Int
	totalBytesSent *expvar.Int
	totalBytesRecv *expvar.Int
	sendRate       *expvar.Float
	recvRate       *expvar.Float
}

func NewRouter(self plan.PeerSpec) (*Router, error) {
	port, err := strconv.Atoi(self.NetAddr.Port)
	if err != nil {
		return nil, err
	}
	return &Router{
		localHost:  self.NetAddr.Host,
		localPort:  uint32(port),
		localSock:  self.SockFile,
		bufferPool: newBufferPool(),     // in-comming messages
		connPool:   newConnectionPool(), // out-going connections

		totalMsgSent:   expvar.NewInt("total_msg_sent"),
		totalMsgRecv:   expvar.NewInt("total_msg_recv"),
		totalBytesSent: expvar.NewInt("total_bytes_sent"),
		totalBytesRecv: expvar.NewInt("total_bytes_recv"),
		sendRate:       expvar.NewFloat("send_bytes_per_sec"),
		recvRate:       expvar.NewFloat("recv_bytes_per_sec"),
	}, nil
}

// getChannel returns the Channel of given Addr
func (r *Router) getChannel(a plan.Addr) (*Channel, error) {
	conn, err := r.connPool.get(a.NetAddr(), r.localHost, r.localPort)
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

func (r *Router) stream(conn net.Conn, remoteNetAddr plan.NetAddr) (int, error) {
	for i := 0; ; i++ {
		var mh messageHeader
		if err := mh.ReadFrom(conn); err != nil {
			return i, err
		}
		// log.Infof("got message header: %s", mh)
		var m Message
		if err := m.ReadFrom(conn); err != nil {
			return i, err
		}
		// log.Infof("got message: %s :: %s", m, string(m.Data))
		a := remoteNetAddr.WithName(string(mh.Name))
		r.bufferPool.require(a) <- &m
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
