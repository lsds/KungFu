package rchannel

import (
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"sync"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Server receives messages from remove endpoints
type Server interface {
	Serve()
	Close()
}

// NewServer creates a new Server
func NewServer(router *Router) (Server, error) {
	tcpServer, err := newTCPServer(router)
	if err != nil {
		return nil, err
	}
	var unixServer *server
	if kc.UseUnixSock {
		var err error
		unixServer, err = newUnixServer(router)
		if err != nil {
			return nil, err
		}
	}
	return &composedServer{
		tcpServer:  tcpServer,
		unixServer: unixServer,
	}, nil
}

type composedServer struct {
	tcpServer  *server
	unixServer *server
}

func (s *composedServer) Serve() {
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
	listener net.Listener
	router   *Router
	unix     bool
}

func newTCPServer(router *Router) (*server, error) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(router.localAddr.Port)))
	log.Debugf("listening: %s", addr)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	return &server{
		listener: listener,
		router:   router,
	}, nil
}

// newUnixServer creates a new Server listening Unix socket
func newUnixServer(router *Router) (*server, error) {
	sockFile := router.localAddr.SockFile()
	cleanSockFile := true
	if cleanSockFile {
		os.Remove(sockFile)
	}
	listener, err := net.ListenUnix("unix", &net.UnixAddr{sockFile, "unix"})
	if err != nil {
		return nil, err
	}
	return &server{
		listener: listener,
		router:   router,
		unix:     true,
	}, nil
}

// Serve starts the server
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
		os.Remove(s.router.localAddr.SockFile())
	}
}

var (
	errNotImplemented          = errors.New("Not Implemented")
	errInvalidConnectionHeader = errors.New("Invalid connection header")
)

func (s *server) handle(conn net.Conn) error {
	defer conn.Close()
	var ch connectionHeader
	if err := ch.ReadFrom(conn); err != nil {
		return err
	}
	remoteNetAddr := plan.NetAddr{
		Host: formatIPv4(ch.SrcIPv4), // formatIPv4 :: uint32 -> str
		Port: ch.SrcPort,
	}
	log.Debugf("got new connection of type %d from: %s", ch.Type, remoteNetAddr)
	switch ConnType(ch.Type) {
	case ConnPing:
		return s.handlePing(remoteNetAddr, conn)
	case ConnControl:
		return s.handleControl(remoteNetAddr, conn)
	case ConnCollective:
		return s.handleCollective(remoteNetAddr, conn)
	case ConnPeerToPeer:
		return s.handlePeerToPeer(remoteNetAddr, conn)
	default:
		return errInvalidConnectionHeader
	}
}

func (s *server) handlePing(remoteNetAddr plan.NetAddr, conn net.Conn) error {
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

func (s *server) handleControl(remoteNetAddr plan.NetAddr, conn net.Conn) error {
	return errNotImplemented
}

func (s *server) handleCollective(remoteNetAddr plan.NetAddr, conn net.Conn) error {
	if n, err := s.router.handle(conn, remoteNetAddr, ConnCollective); err != nil && err != io.EOF {
		return fmt.Errorf("handle error after handled %d messages: %v", n, err)
	}
	return nil
}

func (s *server) handlePeerToPeer(remoteNetAddr plan.NetAddr, conn net.Conn) error {
	if n, err := s.router.handle(conn, remoteNetAddr, ConnPeerToPeer); err != nil && err != io.EOF {
		return fmt.Errorf("handle error after handled %d messages: %v", n, err)
	}
	return nil
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

func formatIPv4(ipv4 uint32) string {
	ip := net.IPv4(byte(ipv4>>24), byte(ipv4>>16), byte(ipv4>>8), byte(ipv4))
	return ip.String()
}
