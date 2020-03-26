package server

import (
	"fmt"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/rchannel/handler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

// Server receives messages from remove endpoints
type Server interface {
	Start() error
	Close()
	SetToken(uint32)
}

// New creates a new Server
func New(endpoint handler.Endpoint) Server {
	tcpServer := newTCPServer(endpoint)
	var unixServer *server
	if kc.UseUnixSock {
		unixServer = newUnixServer(endpoint)
	}
	return &composedServer{
		tcpServer:  tcpServer,
		unixServer: unixServer,
	}
}

type composedServer struct {
	tcpServer  *server
	unixServer *server
}

func (s *composedServer) SetToken(token uint32) {
	for _, srv := range []*server{s.tcpServer, s.unixServer} {
		if srv != nil {
			srv.SetToken(token)
		}
	}
}

func (s *composedServer) Start() error {
	for _, srv := range []*server{s.tcpServer, s.unixServer} {
		if srv != nil {
			if err := srv.Listen(); err != nil {
				return err
			}
		}
	}
	go s.serve()
	return nil
}

func (s *composedServer) serve() {
	var wg sync.WaitGroup
	for _, srv := range []*server{s.tcpServer, s.unixServer} {
		if srv != nil {
			wg.Add(1)
			go func(srv *server) {
				srv.Serve()
				wg.Done()
			}(srv)
		}
	}
	wg.Wait()
}

func (s *composedServer) Close() {
	for _, srv := range []*server{s.tcpServer, s.unixServer} {
		if srv != nil {
			srv.Close()
		}
	}
	log.Debugf("Server Closed")
}

type server struct {
	listen   func() (net.Listener, error)
	listener net.Listener
	self     plan.PeerID
	endpoint handler.Endpoint
	token    uint32
	unix     bool
}

func newTCPServer(endpoint handler.Endpoint) *server {
	return &server{
		listen: func() (net.Listener, error) {
			listenAddr := endpoint.Self()
			listenAddr.IPv4 = 0
			log.Debugf("listening: %s", listenAddr)
			return net.Listen("tcp", listenAddr.String())
		},
		self:     endpoint.Self(),
		endpoint: endpoint,
	}
}

func fileExists(filename string) (bool, time.Duration) {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false, 0
	}
	if err != nil {
		return false, 0
	}
	return true, time.Since(info.ModTime())
}

// newUnixServer creates a new Server listening Unix socket
func newUnixServer(endpoint handler.Endpoint) *server {
	listen := func() (net.Listener, error) {
		sockFile := endpoint.Self().SockFile()
		if ok, age := fileExists(sockFile); ok {
			if age > 0 {
				log.Warnf("%s already exists for %s, trying to remove", sockFile, age)
				if err := os.Remove(sockFile); err != nil {
					utils.ExitErr(err)
				}
			} else {
				utils.ExitErr(fmt.Errorf("can't cleanup socket file: %s", sockFile))
			}
		}
		return net.ListenUnix("unix", &net.UnixAddr{Name: sockFile, Net: "unix"})
	}
	return &server{
		listen:   listen,
		endpoint: endpoint,
		unix:     true,
	}
}

func (s *server) SetToken(token uint32) {
	atomic.StoreUint32(&s.token, token)
}

func (s *server) Listen() error {
	var err error
	s.listener, err = s.listen()
	return err
}

func (s *server) accept() (connection.Connection, error) {
	tcpConn, err := s.listener.Accept()
	if err != nil {
		return nil, err
	}
	conn, err := connection.UpgradeFrom(tcpConn, s.self, atomic.LoadUint32(&s.token))
	if err != nil {
		return nil, err
	}
	return conn, nil
}

func (s *server) Serve() {
	for {
		conn, err := s.accept()
		if err != nil {
			if isNetClosingErr(err) {
				break
			}
			log.Infof("Accept failed: %v", err)
			continue
		}
		go s.handle(conn)
	}
}

// Close closes the listener
func (s *server) Close() {
	// TODO: to be graceful
	s.listener.Close()
	if s.unix {
		os.Remove(s.endpoint.Self().SockFile())
	}
}

func (s *server) handle(conn connection.Connection) {
	defer conn.Close()
	if n, err := s.endpoint.Handle(conn); err != nil {
		log.Warnf("handle conn err: %v after handled %d messages", err, n)
	}
}

// check if error is internal/poll.ErrNetClosing
func isNetClosingErr(err error) bool {
	// file:///$GOROOT/src/internal/poll/fd.go:18:
	// var ErrNetClosing = errors.New("use of closed network connection")
	const msg = `use of closed network connection`
	if e, ok := err.(*net.OpError); ok {
		return msg == e.Err.Error()
	}
	return false
}
