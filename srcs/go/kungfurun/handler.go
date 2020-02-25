package kungfurun

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	rch "github.com/lsds/KungFu/srcs/go/rchannel"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Stage struct {
	Version int
	Cluster plan.PeerList
	Size    int
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

func (s Stage) Eq(t Stage) bool {
	return s.Version == t.Version && s.Cluster.Eq(t.Cluster)
}

type Handler struct {
	self plan.PeerID

	mu       sync.RWMutex
	versions map[int]Stage
	ch       chan Stage
	cancel   context.CancelFunc

	controlHandlers map[string]rch.MsgHandleFunc
}

func (h *Handler) Self() plan.PeerID {
	return h.self
}

func NewHandler(self plan.PeerID, ch chan Stage, cancel context.CancelFunc) *Handler {
	h := &Handler{
		self:            self,
		versions:        make(map[int]Stage),
		ch:              ch,
		cancel:          cancel,
		controlHandlers: make(map[string]rch.MsgHandleFunc),
	}
	h.controlHandlers["update"] = h.handleContrlUpdate
	h.controlHandlers["exit"] = h.handleContrlExit
	return h
}

func (h *Handler) Handle(conn net.Conn, remote plan.NetAddr, t rch.ConnType) error {
	switch t {
	case rch.ConnControl:
		if n, err := rch.Stream(conn, remote, rch.Accept, h.handleControl); err != nil {
			return fmt.Errorf("stream error after handled %d messages: %v", n, err)
		}
		return nil
	default:
		return fmt.Errorf("%v: %s from %s", rch.ErrInvalidConnectionType, t, remote)
	}
}

func (h *Handler) handleControl(name string, msg *rch.Message, conn net.Conn, remote plan.NetAddr) {
	log.Debugf("got control message from %s, name: %s, length: %d", remote, name, msg.Length)
	handle, ok := h.controlHandlers[name]
	if !ok {
		log.Warnf("invalid control messaeg: %s", name)
	}
	handle(name, msg, conn, remote)
}

var errInconsistentUpdate = errors.New("inconsistent update detected")

func (h *Handler) handleContrlUpdate(_name string, msg *rch.Message, _conn net.Conn, remote plan.NetAddr) {
	var s Stage
	if err := s.Decode(msg.Data); err != nil {
		log.Warnf("invalid update message: %v", err)
		return
	}
	s.Size = len(s.Cluster)
	func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		if val, ok := h.versions[s.Version]; ok {
			if !val.Eq(s) {
				utils.ExitErr(errInconsistentUpdate)
			}
			return
		}
		h.versions[s.Version] = s
		h.ch <- s
		log.Debugf("update to v%d with %d peers", s.Version, len(s.Cluster))
	}()
}

func (h *Handler) handleContrlExit(_name string, msg *rch.Message, _conn net.Conn, remote plan.NetAddr) {
	log.Infof("exit control message received.")
	h.cancel()
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	e := json.NewEncoder(w)
	e.SetIndent("", "    ")
	h.mu.RLock()
	defer h.mu.RUnlock()
	e.Encode(h.versions)
}
