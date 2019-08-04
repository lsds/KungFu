package kungfu

import (
	"encoding/json"
	"log"
	"net/http"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type configServer struct {
	configs map[string]string
}

func NewConfigServer(cs *plan.ClusterSpec) http.Handler {
	return &configServer{
		configs: map[string]string{
			kb.ClusterSpecEnvKey: toJSON(cs),
		},
	}
}

func (cs *configServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	log.Printf("%s %s %s", req.Method, req.URL.Path, req.URL.RawQuery)
	val := cs.configs[req.FormValue("name")]
	w.Write([]byte(val))
}

func toJSON(i interface{}) string {
	bs, err := json.Marshal(i)
	if err != nil {
		return ""
	}
	return string(bs)
}
