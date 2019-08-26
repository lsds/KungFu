package sharedvariable

import (
	"log"
	"net/http"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type Server struct {
	m *SharedVariableManager
}

func NewServer() *Server {
	return &Server{
		m: NewSharedVariableManager(),
	}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	log.Printf("%s %s %s", req.Method, req.URL.Path, req.URL.RawQuery)
	name := req.FormValue("name")
	dtype, err := strconv.Atoi(req.FormValue("dtype"))
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}
	count, err := strconv.Atoi(req.FormValue("count"))
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}

	switch req.Method {
	case http.MethodPost:
		s.m.Create(name, count, kb.KungFu_Datatype(dtype))
		break
	case http.MethodGet:
		v, err := s.m.get(name)
		if err != nil {
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if err := v.do(func(data *kb.Buffer) { writeTo(w, data) }); err != nil {
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		break
	case http.MethodPut:
		v, err := s.m.get(name)
		if err != nil {
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if err := v.ReadFrom(req.Body); err != nil {
			log.Printf("v.ReadFrom failed: %v", err)
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		break
	case http.MethodPatch:
		buf := kb.NewBuffer(count, kb.KungFu_Datatype(dtype))
		if err := readBuf(req.Body, buf); err != nil {
			return
		}
		s.m.Add(name, buf, nil)
		break
	default:
		break
	}
}
