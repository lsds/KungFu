package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/rchannel/connection"
	"github.com/lsds/KungFu/srcs/go/rchannel/handler"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Stage struct {
	Version int
	Cluster plan.Cluster
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

	controlHandlers map[string]connection.MsgHandleFunc
	pingHandler     *handler.PingHandler
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
		controlHandlers: make(map[string]connection.MsgHandleFunc),
		pingHandler:     &handler.PingHandler{},
	}
	h.controlHandlers["update"] = h.handleContrlUpdate
	h.controlHandlers["exit"] = h.handleContrlExit
	return h
}

func (h *Handler) Handle(conn connection.Connection) (int, error) {
	switch t := conn.Type(); t {
	case connection.ConnControl:
		return connection.Stream(conn, connection.Accept, h.handleControl)
	case connection.ConnPing:
		return h.pingHandler.Handle(conn)
	default:
		return 0, fmt.Errorf("%v: %s from %s", connection.ErrInvalidConnectionType, t, conn.Src())
	}
}

func (h *Handler) handleControl(name string, msg *connection.Message, conn connection.Connection) {
	log.Debugf("got control message from %s, name: %s, length: %d", conn.Src(), name, msg.Length)
	handle, ok := h.controlHandlers[name]
	if !ok {
		log.Warnf("invalid control messaeg: %s", name)
	}
	handle(name, msg, conn)
}

var errInconsistentUpdate = errors.New("inconsistent update detected")

func (h *Handler) handleContrlUpdate(_name string, msg *connection.Message, _conn connection.Connection) {
	var s Stage
	if err := s.Decode(msg.Data); err != nil {
		log.Warnf("invalid update message: %v", err)
		return
	}
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
		log.Debugf("update to v%d with %s", s.Version, s.Cluster.DebugString())
	}()
}

func (h *Handler) handleContrlExit(_name string, msg *connection.Message, _conn connection.Connection) {
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
