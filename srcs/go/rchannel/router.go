package rchannel

import (
	"expvar"
	"fmt"
	"net"
	"strconv"
	"time"

	"github.com/luomai/kungfu/srcs/go/log"
	"github.com/luomai/kungfu/srcs/go/metrics"
)

type Router struct {
	localPort  uint32
	bufferPool *BufferPool
	connPool   *ConnectionPool

	totalMsgSent   *expvar.Int
	totalMsgRecv   *expvar.Int
	totalBytesSent *expvar.Int
	totalBytesRecv *expvar.Int
	sendRate       *expvar.Float
	recvRate       *expvar.Float
}

func NewRouter(cluster *ClusterSpec) (*Router, error) {
	port, err := strconv.Atoi(cluster.Self.NetAddr.Port)
	if err != nil {
		return nil, err
	}
	return &Router{
		localPort:  uint32(port),
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
func (r *Router) getChannel(a Addr) (*Channel, error) {
	netAddr := net.JoinHostPort(a.Host, a.Port)
	conn, err := r.connPool.get(netAddr, r.localPort)
	if err != nil {
		return nil, err
	}
	return newChannel(a.Name, conn), nil
}

// Send sends Message to given Addr
func (r *Router) Send(a Addr, m Message) error {
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
func (r *Router) Recv(a Addr, m *Message) error {
	// log.Infof("%s::%s(%s)", "Router", "Recv", a)
	// TODO: reduce memory copy
	*m = *<-r.bufferPool.require(a)
	r.totalMsgRecv.Add(1)
	r.totalBytesRecv.Add(int64(m.Length))
	// TODO: add timeout
	return nil
}

func (r *Router) stream(conn net.Conn, remoteNetAddr NetAddr) error {
	for {
		var mh messageHeader
		if err := mh.ReadFrom(conn); err != nil {
			return err
		}
		// log.Infof("got message header: %s", mh)
		var m Message
		if err := m.ReadFrom(conn); err != nil {
			return err
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

	rxRate := rate(rx, d)
	txRate := rate(tx, d)
	r.recvRate.Set(rxRate)
	r.sendRate.Set(txRate)

	const Mi = 1 << 20
	log.Infof("rx_rate: %s, tx_rate: %s", showRate(rxRate), showRate(txRate))
}

func rate(n int64, d time.Duration) float64 {
	return float64(n) / (float64(d) / float64(time.Second))
}

func showRate(r float64) string {
	const Ki = 1 << 10
	const Mi = 1 << 20
	const Gi = 1 << 30
	switch {
	case r > Gi:
		return fmt.Sprintf("%.2f GiB/s", r/float64(Gi))
	case r > Mi:
		return fmt.Sprintf("%.2f MiB/s", r/float64(Mi))
	case r > Ki:
		return fmt.Sprintf("%.2f KiB/s", r/float64(Ki))
	default:
		return fmt.Sprintf("%.2f B/s", r)
	}
}
