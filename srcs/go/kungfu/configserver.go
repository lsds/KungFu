package kungfu

import (
	"encoding/json"
	"log"
	"net/http"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type config struct {
	values map[string]string
}

type configServer struct {
	versions map[string]config
}

func NewConfigServer(cs *plan.ClusterSpec) http.Handler {
	init := config{
		values: map[string]string{
			kb.ClusterSpecEnvKey: toJSON(cs),
		},
	}
	return &configServer{
		versions: map[string]config{
			"": init,
		},
	}
}

func (cs *configServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	log.Printf("%s %s %s", req.Method, req.URL.Path, req.URL.RawQuery)
	version := req.FormValue("version")
	name := req.FormValue("name")
	config, ok := cs.versions[version]
	if !ok {
		http.NotFound(w, req)
		return
	}
	val, ok := config.values[name]
	if !ok {
		http.NotFound(w, req)
		return
	}
	w.Write([]byte(val))
}

func toJSON(i interface{}) string {
	bs, err := json.Marshal(i)
	if err != nil {
		return ""
	}
	return string(bs)
}
