package rchannel

import (
	"fmt"
	"io"
	"net"
	"os"
	"strconv"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

// Server receives messages from remove endpoints
type Server struct {
	listener net.Listener
	router   *Router
	unix     bool
}

// NewServer creates a new Server
func NewServer(router *Router) (*Server, error) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(router.localPort)))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	return &Server{
		listener: listener,
		router:   router,
	}, nil
}

// NewLocalServer creates a new Server listening Unix socket
func NewLocalServer(router *Router) (*Server, error) {
	cleanSockFile := true
	if cleanSockFile {
		os.Remove(router.localSock)
	}
	listener, err := net.ListenUnix("unix", &net.UnixAddr{router.localSock, "unix"})
	if err != nil {
		return nil, err
	}
	return &Server{
		listener: listener,
		router:   router,
		unix:     true,
	}, nil
}

// Serve starts the server
func (s *Server) Serve() {
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
func (s *Server) Close() {
	// TODO: to be graceful
	s.listener.Close()
	if s.unix {
		os.Remove(s.router.localSock)
	}
}

func (s *Server) getRemoveHost(conn net.Conn) string {
	h, _, err := net.SplitHostPort(conn.RemoteAddr().String())
	if err == nil {
		return h
	}
	return s.router.localHost
}

func (s *Server) handle(conn net.Conn) error {
	defer conn.Close()
	var ch connectionHeader
	if err := ch.ReadFrom(conn); err != nil {
		return err
	}
	remoteNetAddr := plan.NetAddr{
		Host: s.getRemoveHost(conn),
		Port: strconv.Itoa(int(ch.Port)),
	}
	log.Debugf("got new connection from: %s", remoteNetAddr)
	if n, err := s.router.stream(conn, remoteNetAddr); err != nil && err != io.EOF {
		return fmt.Errorf("stream error after handled %d messages: %v", n, err)
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
