package rchannel

import (
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

// Server receives messages from remove endpoints
type Server interface {
	Start() error
	Close()
}

// NewServer creates a new Server
func NewServer(endpoint Endpoint) Server {
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
	endpoint Endpoint
	unix     bool
}

func newTCPServer(endpoint Endpoint) *server {
	return &server{
		listen: func() (net.Listener, error) {
			listenAddr := endpoint.Self()
			listenAddr.IPv4 = 0
			log.Debugf("listening: %s", listenAddr)
			return net.Listen("tcp", listenAddr.String())
		},
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
func newUnixServer(endpoint Endpoint) *server {
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

func (s *server) Listen() error {
	var err error
	s.listener, err = s.listen()
	return err
}

func (s *server) Serve() {
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			if isNetClosingErr(err) {
				break
			}
			log.Infof("Accept failed: %v", err)
			continue
		}
		go func(conn net.Conn) {
			if err := s.handle(conn); err != nil {
				log.Warnf("handle conn err: %v", err)
			}
		}(conn)
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

func (s *server) handle(conn net.Conn) error {
	defer conn.Close()
	var ch connectionHeader
	if err := ch.ReadFrom(conn); err != nil {
		return err
	}
	remote := plan.NetAddr{IPv4: ch.SrcIPv4, Port: ch.SrcPort}
	t := ConnType(ch.Type)
	log.Debugf("got new connection of type %s from: %s", t, remote)
	if t == ConnPing {
		return s.handlePing(remote, conn)
	}
	return s.endpoint.Handle(conn, remote, t)
}

func (s *server) handlePing(remote plan.NetAddr, conn net.Conn) error {
	var mh messageHeader
	if err := mh.ReadFrom(conn); err != nil {
		return err
	}
	var empty Message
	if err := empty.ReadFrom(conn); err != nil {
		return err
	}
	if err := mh.WriteTo(conn); err != nil {
		return err
	}
	return empty.WriteTo(conn)
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
