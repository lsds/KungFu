package kungfuprun

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
)

type Stage struct {
	Checkpoint string
	Cluster    plan.PeerList
}

func (s Stage) Encode() []byte {
	b := &bytes.Buffer{}
	json.NewEncoder(b).Encode(s)
	return b.Bytes()
}

func (s *Stage) Decode(bs []byte) error {
	b := bytes.NewBuffer(bs)
	return json.NewDecoder(b).Decode(s)
}

type Handler struct {
	self plan.PeerID

	mu          sync.Mutex
	checkpoints map[string]Stage
	ch          chan Stage
}

func (h *Handler) Self() plan.PeerID {
	return h.self
}

func NewHandler(self plan.PeerID, ch chan Stage) *Handler {
	return &Handler{
		self:        self,
		checkpoints: make(map[string]Stage),
		ch:          ch,
	}
}

func (h *Handler) Handle(conn net.Conn, remote plan.NetAddr, t rch.ConnType) error {
	switch t {
	case rch.ConnControl:
		if n, err := rch.Stream(conn, remote, rch.Accept, h.handleControl); err != nil {
			return fmt.Errorf("stream error after handled %d messages: %v", n, err)
		}
		return nil
	default:
		return rch.ErrInvalidConnectionType
	}
}

func (h *Handler) handleControl(name string, msg *rch.Message, _conn net.Conn, remote plan.NetAddr) {
	log.Debugf("got control message from %s, name: %s, length: %d", remote, name, msg.Length)
	if name == "update" {
		var s Stage
		if err := s.Decode(msg.Data); err != nil {
			log.Warnf("invalid update message: %v", err)
			return
		}
		func() {
			h.mu.Lock()
			defer h.mu.Unlock()
			if _, ok := h.checkpoints[s.Checkpoint]; ok {
				// FIXME: check content
				return
			}
			h.checkpoints[s.Checkpoint] = s
			h.ch <- s
			log.Debugf("update to %s with %d peers", s.Checkpoint, len(s.Cluster))
		}()
	}
}
