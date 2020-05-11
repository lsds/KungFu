package server

import (
	"os"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/utils"
)

// Server receives messages from remove endpoints
type Server interface {
	Start() error
	Close()
	SetToken(uint32)
}

// New creates a new Server
func New(self plan.PeerID, handler connection.Handler, useUnixSock bool) *composedServer {
	tcpServer := newTCPServer(self, handler)
	var unixServer *server
	if useUnixSock {
		unixServer = newUnixServer(self, handler)
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

func (s *composedServer) listen() error {
	for _, srv := range []*server{s.tcpServer, s.unixServer} {
		if srv != nil {
			if err := srv.Listen(); err != nil {
				return err
			}
		}
	}
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

func (s *composedServer) ListenAndServe() error {
	if err := s.listen(); err != nil {
		return err
	}
	s.serve()
	return nil
}

func (s *composedServer) Start() error {
	if err := s.listen(); err != nil {
		return err
	}
	go s.serve()
	utils.Trap(func(os.Signal) { s.Close() })
	return nil
}

func (s *composedServer) Close() {
	for _, srv := range []*server{s.tcpServer, s.unixServer} {
		if srv != nil {
			srv.Close()
		}
	}
	log.Debugf("Server Closed")
}
