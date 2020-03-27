package server

import (
	"fmt"
	"net"
	"os"
	"sync/atomic"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type server struct {
	listen   func() (net.Listener, error)
	listener net.Listener
	self     plan.PeerID
	handler  connection.Handler
	token    uint32
	unix     bool
}

func newTCPServer(self plan.PeerID, handler connection.Handler) *server {
	return &server{
		listen: func() (net.Listener, error) {
			listenAddr := self.ListenAddr(false)
			log.Debugf("listening: %s", listenAddr)
			return net.Listen("tcp", listenAddr.String())
		},
		self:    self,
		handler: handler,
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
func newUnixServer(self plan.PeerID, handler connection.Handler) *server {
	listen := func() (net.Listener, error) {
		sockFile := self.SockFile()
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
		listen:  listen,
		self:    self,
		handler: handler,
		unix:    true,
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
		os.Remove(s.self.SockFile())
	}
}

func (s *server) handle(conn connection.Connection) {
	defer conn.Close()
	if n, err := s.handler.Handle(conn); err != nil {
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
