package rchannel

import (
	"net"
	"strconv"

	"github.com/luomai/kungfu/srcs/go/log"
)

// Server receives messages from remove endpoints
type Server struct {
	listener net.Listener
	router   *Router
}

const (
	defaultPort = 10000 + 1
)

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

// ListenAndServe starts the server
func (s *Server) ListenAndServe() {
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
}

func (s *Server) handle(conn net.Conn) error {
	defer conn.Close()
	h, _, err := net.SplitHostPort(conn.RemoteAddr().String())
	if err != nil {
		return err
	}
	var ch connectionHeader
	if err := ch.ReadFrom(conn); err != nil {
		return err
	}
	remoteNetAddr := NetAddr{
		Host: h,
		Port: strconv.Itoa(int(ch.Port)),
	}
	log.Infof("got new connection from: %s", remoteNetAddr)
	return s.router.stream(conn, remoteNetAddr)
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
